import os
import pandas as pd
from transformers import PreTrainedModel, AutoTokenizer, AutoConfig
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

class CausalLMModelForOV(CausalLMModelForOnnxGeneration):
    def __init__(
        self, model_path, config=None, threads: int = 0
    ):
        if config is None:
            config = AutoConfig.from_pretrained(model_path)
        PreTrainedModel.__init__(self, config)

        self.core = Core()
        self.net = self.core.read_model(model='/home/luocheng/openvino/hacked/gpt_neox.xml')
        #serialize(self.net, "1.xml", "1.bin")
        self.batch = 2
        self.net.reshape({'input_ids': [self.batch, 300], #[2, -1], # [-1, -1],
                          #'past_key_values': [32,2,self.batch,32,-1,80] #[12,2,2,12,-1,64] #[12,2,-1,12,-1,64]
                          })
        config = {'PERFORMANCE_HINT': '',
            'NUM_STREAMS': '1',
            'INFERENCE_PRECISION_HINT': 'f32',
            'CPU_RUNTIME_CACHE_CAPACITY': '5000000',
            'AFFINITY': 'CORE',
            #'PERFORMANCE_HINT_NUM_REQUESTS': '2'
            #'ENFORCE_BF16': 'YES'
            'INFERENCE_NUM_THREADS': '56' #'64'
            }

        self.exec_net300 = self.core.compile_model(self.net, 'CPU', config)
        self.net.reshape({'input_ids': [self.batch, 1], #[2, -1], # [-1, -1],
                          #'past_key_values': [32,2,self.batch,32,-1,80] #[12,2,2,12,-1,64] #[12,2,-1,12,-1,64]
                          })
        self.exec_net1 = self.core.compile_model(self.net, 'CPU', config)
        self.req300 = self.exec_net300.create_infer_request()
        self.req1 = self.exec_net1.create_infer_request()
        self.idx = 0
        self.idx_test = 0
        self.stat = {
            'init': 0,
            'infer_1x300': 0,
            'infer_1x1': 0,
            'post': 0,
            'times': 0
        }

        # self.net.reshape({'input_ids': [2, 1], # [-1, -1],
        #                   'past_key_values': [12,2,2,12,-1,64] #[12,2,-1,12,-1,64]
        #                   })
        # self.exec_net2 = self.core.compile_model(self.net, 'CPU', config)
        # model = self.exec_net2.get_runtime_model()
        # serialize(model, 'exec2.xml', 'exec2.bin')
        # self.req2 = AsyncInferQueue(self.exec_net2, self.nireq) #self.exec_net1.create_infer_request()

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

        if input_ids.shape[1] == 300:
            past_key_num = np.array([0,], dtype=np.int64)
            self.idx = 0
        else:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            past_key_num = np.array([300 + self.idx,], dtype=np.int64)
            self.idx += 1
        input_ids_np = input_ids.cpu().numpy()
        #np.save('input_ids.npy', input_ids_np)
        inputs = {
            0: Tensor(input_ids_np),
            1: Tensor(past_key_num),
        }
        #print(f'cost0 {time.time() - beg} with {inputs1[0].shape}')
        self.stat['init'] += time.time() - beg
        beg = time.time()
        # self.req1.set_tensors(inputs)
        # self.req1.infer()
        if input_ids.shape[1] == 300:
            self.req300.set_input_tensors(inputs)
            self.req300.infer()
            self.stat['infer_1x300'] += time.time() - beg
            logits, = self.req300.outputs
        else:
            self.req1.set_input_tensors(inputs)
            self.req1.infer()
            cost = time.time() - beg
            self.stat['infer_1x1'] += cost
            #print(f'{self.idx} {cost}')
            logits, = self.req1.outputs

        #print(f'cost1 {time.time() - beg} with {inputs1[0].shape}')
        beg = time.time()
        x = torch.from_numpy(logits.data)
        ref = np.load(f'lm_logits{self.idx_test}.npy', allow_pickle=True)
        if not np.allclose(x, ref, 0.01, 0.01):
            print(f'error {self.idx_test}')
        self.idx_test += 1
        #print(f'cost2 {time.time() - beg} with {inputs1[0].shape}')
        self.stat['post'] += time.time() - beg
        self.stat['times'] += 1

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=x,
            past_key_values=None,
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
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # if self.idx // 100 == 0:
        #     # only last token for inputs_ids if past is defined in kwargs
        #     input_ids = input_ids[:, -1].unsqueeze(-1)
        #     if token_type_ids is not None:
        #         token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = None
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "token_type_ids": token_type_ids,
        }


device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('model_big')
model = CausalLMModelForOV.from_pretrained('model_big')

# workaround model.device check begin
old_get_parameter_device = transformers.modeling_utils.get_parameter_device
def my_get_parameter_device(parameter):
    if parameter == model:
        return torch.device("cpu")
    else:
        return old_get_parameter_device(parameter)
transformers.modeling_utils.get_parameter_device = my_get_parameter_device
# workaround model.device check end

tokenizer.pad_token = tokenizer.eos_token
df = pd.read_json('results/a100-asparagus-infers.jsonl', lines=True)
f = open('ov-results-attn.txt', 'w')
for j, i in enumerate(df.prompt.iloc[:5]):
    input_ids = tokenizer.encode(i, return_tensors='pt', add_special_tokens=False)
    if len(input_ids[0]) >= 300:
        input_ids = input_ids[:, -300:]
    beg = time.time()
    outputs = model.generate(input_ids.to(device), pad_token_id=tokenizer.eos_token_id,
                   num_beams=2, max_new_tokens=100, temperature=1.0)
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
