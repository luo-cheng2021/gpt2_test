import torch
import os
from openvino.runtime import Core, Model, Tensor, PartialShape, serialize, AsyncInferQueue
from openvino.runtime.passes import Manager
from openvino.runtime.passes import VisualizeTree
import time
import numpy as np
from torch import nn

from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, serialize
from openvino.runtime.op import util as op_util
from openvino.runtime import opset10 as opset
from openvino.runtime.passes import Manager

LAYER_NORM_EPS = 1e-5
# test mvn int8  
def gen_ov_i8(qkvs, is_ref=False):
    def make_model():
        # [batch, tokens, 2560]
        qkv = opset.parameter([-1, -1, 2560], Type.f32, name='qkv')
        embed_out_weight_const = np.ones((64, 2560), dtype=np.float32)
        embed_out_weight_const[:,0::1] = -1
        embed_out_weight = opset.constant(embed_out_weight_const, Type.f32)
        if not is_ref:
            q_d = opset.fake_quantize(qkv, -1, 1, -1, 1, 255, "NUMPY", name=f'fq_input_0')
            q_w = opset.fake_quantize(embed_out_weight, -1, 1, -1, 1, 255, "NUMPY", name=f'fq_weights_1')
            embed_out = opset.matmul(q_d, q_w, transpose_a=False,transpose_b=True, name='fc')
            gelu = opset.gelu(embed_out, approximation_mode='erf', name=f'gelu')
            fq = opset.fake_quantize(gelu, -1, 1, -1, 1, 255, "NUMPY", name=f'fq_result')
            fq_result = opset.convert(fq, np.int8)
        else:
            # q_d = opset.unsqueeze(qkv, 0)
            # q_d = opset.fake_quantize(q_d, -1, 1, -1, 1, 255, "NUMPY", name=f'fq_input_0')
            # q_d = opset.squeeze(q_d, [0])
            q_d = opset.clamp(qkv, -1, 1, name='clamp')
            # q_d = opset.multiply(q_d, np.array([127], dtype=np.float32))
            # q_d = opset.convert(q_d, np.int8)
            # q_d = opset.convert(q_d, np.float32)
            # q_d = opset.multiply(q_d, np.array([1.0/127], dtype=np.float32))
            q_w = opset.fake_quantize(embed_out_weight, -1, 1, -1, 1, 255, "NUMPY", name=f'fq_weights_1')
            embed_out = opset.matmul(q_d, q_w, transpose_a=False,transpose_b=True, name='fc')
            embed_out = opset.gelu(embed_out, approximation_mode='erf', name=f'gelu')
            embed_out = opset.unsqueeze(embed_out, 0, name='unsqueeze')
            #fq = opset.fake_quantize(embed_out, -1, 1, -1, 1, 255, "NUMPY", name=f'fq_result')
            #fq = opset.multiply(fq, np.array([127], dtype=np.float32), name='mul')
            embed_out = opset.clamp(embed_out, -1, 1, name='clamp1')
            fq = opset.multiply(embed_out, np.array([127], dtype=np.float32))
            fq_result = opset.squeeze(fq, [0], name='squeeze')

        return Model([fq_result], [qkv])
    net = make_model()
    core = Core()
    config = {'PERFORMANCE_HINT': '',
            'NUM_STREAMS': '1',
            'ENFORCE_BF16': 'YES'
            }
    model = core.compile_model(net, 'CPU', config)
    req = model.create_infer_request()
    results = []
    for (i, qkv) in enumerate(qkvs):
        inputs = {
            0: Tensor(qkv),
        }
        req.set_input_tensors(inputs)
        req.infer()
        r, = req.outputs
        results.append(np.array(r.data).astype(np.int8))
    #m = model.get_runtime_model()
    #m = model.net
    #serialize(m, 'ov-fc.xml', 'ov-fc.bin')

    return results

def test_fc():
    qkvs = [
            np.random.random(size=[2, 900, 2560]).astype(np.float32)*10,
            np.random.random(size=[2, 1, 2560]).astype(np.float32)*10,
            ]
    qkvs[0][:,:,0::3] = -1
    qkvs[0][:,:,1::3] = +1
    qkvs[0][:,:,2::3] = 0
    
    print('test ref...')
    ref_results = gen_ov_i8(qkvs, True)
    print('\ntest cur...')
    cur_results = gen_ov_i8(qkvs, False)

    for (i, ref) in enumerate(ref_results):
        if not np.allclose(ref, cur_results[i], rtol=0.001, atol=0.01):
            print('ref:\n', ref, '\ncur:\n', cur_results[i])
            print(f'error: idx {i} not close')
    else:
        print('done')

test_fc()
