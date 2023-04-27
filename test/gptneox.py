import torch
from openvino.runtime import Core, Model, Tensor, PartialShape, serialize, AsyncInferQueue
from openvino.runtime.passes import Manager
from openvino.runtime.passes import VisualizeTree
import os
import numpy as np
from torch import nn

from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, serialize
from openvino.runtime.op import util as op_util
from openvino.runtime import opset10 as opset
from openvino.runtime.passes import Manager

num_attention_heads = 32
head_size = 80
rotary_ndims = 20

# copy from transformers/models/gpt_neox/modeling_gpt_neox.py
# qkv: [batch, seq_len, (np * 3 * head_size)]
# layer_past: 
class GPTNeoXAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims, config.max_position_embeddings, base=config.rotary_emb_base
        )
        self.norm_factor = torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(torch.get_default_dtype())
        #self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        #self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        qkv,
        #hidden_states,
        attention_mask,
        head_mask=None,
        layer_past=None,
        use_cache=False,
        output_attentions=False,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        # qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[-2]
            seq_len += offset
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, offset=offset)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        #attn_output = self.dense(attn_output)

        outputs = (attn_output, present)

        return outputs

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        # -> [bs, seq_len, hidden_size]
        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=(torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor),
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


def attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(~ltor_mask, torch.finfo(attention_scores.dtype).min)
    return attention_scores


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached[:seq_len, ...].to(x.device), self.sin_cached[:seq_len, ...].to(x.device)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[..., offset : q.shape[-2] + offset, :]
    sin = sin[..., offset : q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

HEAD_NUM = 12 #32
SIZE_PER_HEAD = 80
HIDDEN_SIZE = HEAD_NUM * SIZE_PER_HEAD
MAX_POSITION_EMBEDDINGS = 1024 #2048
ROTARY_EMB_BASE = 10000
ROTARY_PCT = 0.25
MAX_SEQ_LEN = 1024

def gen_ref_gpt(qkvs, beam_idxs, attn_masks):
    class FakeConfig:
        def __init__(self):
            self.num_attention_heads = HEAD_NUM
            self.hidden_size = HIDDEN_SIZE
            self.rotary_pct = ROTARY_PCT
            self.max_position_embeddings = MAX_POSITION_EMBEDDINGS
            self.rotary_emb_base = ROTARY_EMB_BASE
    config = FakeConfig()
    net = GPTNeoXAttention(config)
    net = net.to(dtype=torch.bfloat16)
    seq_offset = 0
    results = []
    present = None
    with torch.cpu.amp.autocast():    
        for (i, qkv_) in enumerate(qkvs):
            if present:
                layer_past = [p.index_select(0, torch.tensor(beam_idxs[i])) for p in present]
            else:
                layer_past = None
            qkv = torch.from_numpy(qkv_).to(torch.bfloat16)
            mask = torch.tensor(attn_masks[i][:,:qkv.shape[1]])
            mask = mask[:, None,:, None]
            attention_mask = (1.0 - mask) * torch.finfo(qkv.dtype).min
            attn_output, present = net.forward(
                qkv,
                attention_mask,
                layer_past=layer_past,
                use_cache=True,
            )
            seq_offset += qkv.shape[1]
            results.append(attn_output.clone().to(dtype=torch.float32).detach().numpy())
    return results

def gen_ov_gpt_bf16(qkvs, beam_idxs, attn_masks):
    def make_model():
        # [batch, tokens, 32*80*3]
        qkv = opset.parameter([-1, -1, HEAD_NUM * SIZE_PER_HEAD * 3], Type.f32, name='qkv')
        past_keys_num = opset.parameter([1,], Type.i64, name='past_keys_num')
        beam_idx = opset.parameter([-1,], Type.i64, name='beam_idx')
        attn_mask = opset.parameter([-1, MAX_SEQ_LEN], Type.i64, name='attn_mask')

        # custom op
        attn_output = opset.gpt_neox_attn(qkv, past_keys_num, beam_idx, attn_mask,
                layer_num=0, head_num=HEAD_NUM, size_per_head=SIZE_PER_HEAD, hidden_size=HIDDEN_SIZE, max_position_embeddings=MAX_POSITION_EMBEDDINGS,
                rotary_emb_base=ROTARY_EMB_BASE, rotary_pct=ROTARY_PCT, max_seq_len=MAX_SEQ_LEN, name=f'/model/gpt_neox/layers.0/attention/attn')
        return Model([attn_output], [qkv, past_keys_num, beam_idx, attn_mask])
    net = make_model()
    core = Core()
    config = {'PERFORMANCE_HINT': '',
            'NUM_STREAMS': '1',
            'ENFORCE_BF16': 'YES'
            }
    model = core.compile_model(net, 'CPU', config)
    req = model.create_infer_request()
    results = []
    seq_offset = 0
    for (i, qkv) in enumerate(qkvs):
        attention_mask_np = attn_masks[i]
        past_key_num = np.array([seq_offset], dtype=np.int64)
        seq_offset += qkv.shape[1]
        beam_idx_np = np.array(beam_idxs[i], dtype=np.int64)
        inputs = {
            0: Tensor(qkv),
            1: Tensor(past_key_num),
            2: Tensor(beam_idx_np),
            3: Tensor(attention_mask_np)
        }
        req.set_input_tensors(inputs)
        req.infer()
        r, = req.outputs
        results.append(np.array(r.data))
    return results

# test matmul with s8 input and whole output type is s8
def gen_ov_gpt_i8(qkvs, beam_idxs, attn_masks, is_ref=False):
    def make_model():
        # [batch, tokens, 32*80*3]
        qkv = opset.parameter([-1, -1, HEAD_NUM * SIZE_PER_HEAD * 3], Type.f32, name='qkv')
        past_keys_num = opset.parameter([1,], Type.i64, name='past_keys_num')
        beam_idx = opset.parameter([-1,], Type.i64, name='beam_idx')
        attn_mask = opset.parameter([-1, MAX_SEQ_LEN], Type.i64, name='attn_mask')

        # custom op
        requant = 10.0
        if not is_ref:
            q_quant, k_quant, qk_quant, v_quant = 127.0, 127.0, 255.0, 127.0
            attn_output = opset.gpt_neox_attn(qkv, past_keys_num, beam_idx, attn_mask,
                    layer_num=0, head_num=HEAD_NUM, size_per_head=SIZE_PER_HEAD, hidden_size=HIDDEN_SIZE, max_position_embeddings=MAX_POSITION_EMBEDDINGS,
                    rotary_emb_base=ROTARY_EMB_BASE, rotary_pct=ROTARY_PCT, max_seq_len=MAX_SEQ_LEN,
                    q_quant=q_quant, k_quant=k_quant, qk_quant=qk_quant, v_quant=v_quant,
                    name=f'/model/gpt_neox/layers.0/attention/attn')
            fq = opset.fake_quantize(attn_output, -127/requant, 127/requant, -127/requant, 127/requant, 255, "NUMPY", name=f'fq_input_0')
            fq_result = opset.convert(fq, np.int8)
        else:
            q_quant, k_quant, qk_quant, v_quant = 0.0, 0.0, 0.0, 0.0
            attn_output = opset.gpt_neox_attn(qkv, past_keys_num, beam_idx, attn_mask,
                    layer_num=0, head_num=HEAD_NUM, size_per_head=SIZE_PER_HEAD, hidden_size=HIDDEN_SIZE, max_position_embeddings=MAX_POSITION_EMBEDDINGS,
                    rotary_emb_base=ROTARY_EMB_BASE, rotary_pct=ROTARY_PCT, max_seq_len=MAX_SEQ_LEN,
                    q_quant=q_quant, k_quant=k_quant, qk_quant=qk_quant, v_quant=v_quant,
                    name=f'/model/gpt_neox/layers.0/attention/attn')
            #attn_output = opset.clamp(attn_output, -12.7, 12.7, name='clamp1')
            fq_result = opset.multiply(attn_output, np.array([requant], dtype=np.float32))
            #fq_result = opset.fake_quantize(attn_output, -127, 127, -127, 127, 255, "NUMPY", name=f'fq_input_0')

        return Model([fq_result], [qkv, past_keys_num, beam_idx, attn_mask])
    net = make_model()
    core = Core()
    config = {'PERFORMANCE_HINT': '',
            'NUM_STREAMS': '1',
            'ENFORCE_BF16': 'YES'
            }
    model = core.compile_model(net, 'CPU', config)
    req = model.create_infer_request()
    results = []
    seq_offset = 0
    for (i, qkv) in enumerate(qkvs):
        attention_mask_np = attn_masks[i]
        past_key_num = np.array([seq_offset], dtype=np.int64)
        seq_offset += qkv.shape[1]
        beam_idx_np = np.array(beam_idxs[i], dtype=np.int64)
        inputs = {
            0: Tensor(qkv),
            1: Tensor(past_key_num),
            2: Tensor(beam_idx_np),
            3: Tensor(attention_mask_np)
        }
        req.set_input_tensors(inputs)
        req.infer()
        r, = req.outputs
        results.append((np.array(r.data)+0.5).astype(np.int8))
    return results

def test_gpt_bf16():
    print('testing bf16...')
    qkvs = [
            np.random.random(size=[2, 900, HEAD_NUM * SIZE_PER_HEAD * 3]).astype(np.float32),
            np.random.random(size=[2, 1, HEAD_NUM * SIZE_PER_HEAD * 3]).astype(np.float32),
            np.random.random(size=[2, 1, HEAD_NUM * SIZE_PER_HEAD * 3]).astype(np.float32),
            np.random.random(size=[2, 1, HEAD_NUM * SIZE_PER_HEAD * 3]).astype(np.float32),
            ]
    # for i in range(0):
    #     # qkvs[i][:,:,0::3] = -1
    #     # qkvs[i][:,:,1::3] = +1
    #     # qkvs[i][:,:,2::3] = 0
    #     qkvs[i][:,:,0::2] = -0.5
    #     qkvs[i][:,:,1::2] = +0.5
    #     #qkvs[i][:,:,2::3] = 0
    beam_idxs = [[-1, -1],
                 [1, 0],
                 [1, 1],
                 [0, 0]]
    attn_masks = [np.ones([2, 1024], dtype=np.int64),
                  ] * len(beam_idxs)
    ref_results = gen_ref_gpt(qkvs, beam_idxs, attn_masks)
    ov_results = gen_ov_gpt_bf16(qkvs, beam_idxs, attn_masks)
    for (i, ref) in enumerate(ref_results):
        if not np.allclose(ref, ov_results[i], rtol=0.01, atol=0.01):
            print(f'error: idx {i} not close')
    else:
        print('done')

def test_gpt_i8():
    print('testing i8...')
    qkvs = [
            np.random.randint(low=-1, high=2, size=[2, 100, HEAD_NUM * SIZE_PER_HEAD * 3]).astype(np.float32),
            np.random.randint(low=-1, high=2, size=[2, 1, HEAD_NUM * SIZE_PER_HEAD * 3]).astype(np.float32),
            np.random.randint(low=-1, high=2, size=[2, 1, HEAD_NUM * SIZE_PER_HEAD * 3]).astype(np.float32),
            np.random.randint(low=-1, high=2, size=[2, 1, HEAD_NUM * SIZE_PER_HEAD * 3]).astype(np.float32),
            # np.random.random(size=[2, 100, HEAD_NUM * SIZE_PER_HEAD * 3]).astype(np.float32),
            # np.random.random(size=[2, 1, HEAD_NUM * SIZE_PER_HEAD * 3]).astype(np.float32),
            # np.random.random(size=[2, 1, HEAD_NUM * SIZE_PER_HEAD * 3]).astype(np.float32),
            # np.random.random(size=[2, 1, HEAD_NUM * SIZE_PER_HEAD * 3]).astype(np.float32),
            ]
    for i in range(len(qkvs)):
        pass
        # qkvs[i][:,:,0::3] = -1
        # qkvs[i][:,:,1::3] = +1
        # qkvs[i][:,:,2::3] = 0
    beam_idxs = [[-1, -1],
                 [1, 0],
                 [1, 1],
                 [0, 0]]
    attn_masks = [np.ones([2, 1024], dtype=np.int64),
                  ] * len(beam_idxs)
    for i in range(len(qkvs)):
        #np.save(f'xx{i}.npy', qkvs[i], allow_pickle=True)
        #qkvs[i] = np.load(f'xx{i}.npy', allow_pickle=True)
        pass
    ref_results1 = gen_ref_gpt(qkvs, beam_idxs, attn_masks)
    ref_results = gen_ov_gpt_i8(qkvs, beam_idxs, attn_masks, True)
    ov_results = gen_ov_gpt_i8(qkvs, beam_idxs, attn_masks, False)
    for (i, ref) in enumerate(ref_results):
        #if not np.allclose(ref, ov_results[i], rtol=0.001, atol=0.01):
        if (np.abs(ref_results[i]- ov_results[i])>2).any():
            print(f'error: idx {i} not close')
    else:
        print('done')

test_gpt_bf16()
test_gpt_i8()