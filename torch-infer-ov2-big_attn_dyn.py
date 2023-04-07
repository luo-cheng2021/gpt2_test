import os
import pandas as pd
from transformers import PreTrainedModel, AutoTokenizer, AutoConfig
from transformers.tokenization_utils_base import PaddingStrategy
import torch
import time
from fastgpt.fastgpt import CausalLMModelForOnnxGeneration
import transformers

from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import onnxruntime
from typing import Optional, Tuple
import numpy as np
from fastgpt.fastgpt.onnx_exporter_big import transformers_onnx_pipeline

from openvino.runtime import Core, Model, Tensor, PartialShape, serialize, AsyncInferQueue
from openvino.runtime.passes import Manager
from openvino.runtime.passes import VisualizeTree
import argparse

OV_MODEL_PATH = '/home/xiping/luocheng/openvino/hacked/gpt_neox.xml'
PYTORCH_MODEL_PATH = 'model_big'
FIRST_SIZE = 900
class CausalLMModelForOV(CausalLMModelForOnnxGeneration):
    beam_idx = None
    def __init__(
        self, model_path, config=None, threads: int = 0
    ):
        if config is None:
            config = AutoConfig.from_pretrained(model_path)
        PreTrainedModel.__init__(self, config)

        self.core = Core()
        self.net = self.core.read_model(model=OV_MODEL_PATH)
        #serialize(self.net, "1.xml", "1.bin")
        self.batch = 2
        self.net.reshape({'input_ids': [self.batch, [1, FIRST_SIZE]], #[2, -1], # [-1, -1],
                          'beam_idx': [self.batch],
                          'attn_mask': [self.batch, 1024]
                          })
        config = {'PERFORMANCE_HINT': '',
            'NUM_STREAMS': '1',
            #'INFERENCE_PRECISION_HINT': 'f32',
            'CPU_RUNTIME_CACHE_CAPACITY': '5000000',
            'AFFINITY': 'CORE',
            #'PERFORMANCE_HINT_NUM_REQUESTS': '2'
            'ENFORCE_BF16': 'YES'
            #'INFERENCE_NUM_THREADS': '56' #'64'
            }

        self.exec_net300 = self.core.compile_model(self.net, 'CPU', config)
        self.req300 = self.exec_net300.create_infer_request()
        self.idx = 0
        self.idx_test = 0
        self.first_token = True
        self.stat = {
            'init': 0,
            'infer_1x300': 0,
            'infer_1x1': 0,
            'post': 0,
            'times': 0
        }

    @classmethod
    def from_pretrained(cls, model_name_path: str, threads=0):
        return cls(model_path=model_name_path, threads=threads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        beg = time.time()

        if self.first_token:
            past_key_num = np.array([0,], dtype=np.int64)
            self.idx = 0
            self.seq_offset = input_ids.shape[1]
        else:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            past_key_num = np.array([self.seq_offset + self.idx,], dtype=np.int64)
            self.idx += 1
        input_ids_np = input_ids.cpu().numpy()
        if CausalLMModelForOV.beam_idx is None:
            beam_idx_np = np.zeros(input_ids.shape[0], dtype=np.int64)
        else:
            beam_idx_np = CausalLMModelForOV.beam_idx.cpu().numpy()
        #np.save('input_ids.npy', input_ids_np)
        attention_mask_np = np.zeros([input_ids_np.shape[0], 1024], dtype=np.int64)
        attention_mask_org = attention_mask.cpu().numpy()
        attention_mask_np[:,:attention_mask.shape[1]] = attention_mask_org
        #np.save('attn.npy', attention_mask_np)
        inputs = {
            0: Tensor(input_ids_np),
            1: Tensor(past_key_num),
            2: Tensor(beam_idx_np),
            3: Tensor(attention_mask_np)
        }
        self.stat['init'] += time.time() - beg
        beg = time.time()
        self.req300.set_input_tensors(inputs)
        if self.first_token:
            self.req300.infer()
            self.stat['infer_1x300'] += time.time() - beg
            logits, = self.req300.outputs
        else:
            self.req300.infer()
            
            self.stat['infer_1x1'] += time.time() - beg
            logits, = self.req300.outputs

        beg = time.time()
        x = torch.from_numpy(logits.data)
        self.stat['post'] += time.time() - beg
        self.stat['times'] += 1
        self.first_token = False

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=x,
            past_key_values='not none',
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    @staticmethod
    def _reorder_cache(
        past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        CausalLMModelForOV.beam_idx = beam_idx
        return 'not none'
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # if self.idx // 100 == 0:
        #     # only last token for inputs_ids if past is defined in kwargs
        #     input_ids = input_ids[:, -1].unsqueeze(-1)
        #     if token_type_ids is not None:
        #         token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = None #kwargs.get("position_ids", None)

        # if attention_mask is not None and position_ids is None:
        #     # create position_ids on the fly for batch generation
        #     position_ids = attention_mask.long().cumsum(-1) - 1
        #     position_ids.masked_fill_(attention_mask == 0, 1)
        #     if past_key_values:
        #         position_ids = position_ids[:, -1].unsqueeze(-1)
        # else:
        #     position_ids = None

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(PYTORCH_MODEL_PATH)
    model = CausalLMModelForOV.from_pretrained(PYTORCH_MODEL_PATH)

    # workaround model.device check begin
    old_get_parameter_device = transformers.modeling_utils.get_parameter_device
    def my_get_parameter_device(parameter):
        if parameter == model:
            return torch.device("cpu")
        else:
            return old_get_parameter_device(parameter)
    transformers.modeling_utils.get_parameter_device = my_get_parameter_device
    # workaround model.device check end

    tokenizer.padding_side = 'left'
    df = pd.read_json('results/a100-asparagus-infers.jsonl', lines=True)
    f = open('ov-results-attn.txt', 'w')
    for j, i in enumerate(df.prompt.iloc[:5]):
        input_ids = tokenizer.encode(i, return_tensors='pt', add_special_tokens=False)
        if len(input_ids[0]) >= 900:
            input_ids = input_ids[:, -900:]
        model.first_token = True
        beg = time.time()
        outputs = model.generate(input_ids.to(device), pad_token_id=tokenizer.pad_token_id,
                    num_beams=2, max_new_tokens=90, temperature=1.0)
        end = time.time()
        x = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        f.write('\n'.join(x))
        f.write(f'\n{j} ==============================\n')
        print(f'{j} cost {end-beg:.2f} sec, stat {model.stat}')
        model.stat = {
            'init': 0,
            'infer_1x300': 0,
            'infer_1x1': 0,
            'post': 0,
            'times': 0
        }

    f.close()
    #m = model.exec_net300.get_runtime_model()
    #serialize(m, 'ov-special.xml', 'ov-special.bin')

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("org_model_path")
    parser.add_argument("ov_model_path")
    args = parser.parse_args()
    OV_MODEL_PATH = args.ov_model_path
    PYTORCH_MODEL_PATH = args.org_model_path
    main()