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
        if not is_ref:
            mvn = opset.mvn(qkv, axes=[-1], normalize_variance=True, eps=LAYER_NORM_EPS, eps_mode="inside_sqrt", name='mvn')
            fq = opset.fake_quantize(mvn, -127, 127, -127, 127, 255, "NUMPY", name=f'fq_input_0')
            fq_result = opset.convert(fq, np.int8)
        else:
            mvn_ref = opset.mvn(qkv, axes=[-1], normalize_variance=True, eps=LAYER_NORM_EPS, eps_mode="inside_sqrt", name='mvn_ref')
            fq_result = opset.fake_quantize(mvn_ref, -127, 127, -127, 127, 255, "NUMPY", name=f'fq_input_0')
        
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
        results.append(np.array(r.data))
    #m = model.get_runtime_model()
    #m = model.net
    #serialize(m, 'ov-mvn.xml', 'ov-mvn.bin')

    return results

LEN=2560
# test mvn int8  
def gen_ov_mvn_custom(qkvs, weight, bias, is_i8, is_ref=False):
    def make_model():
        # [batch, tokens, 2560]
        qkv = opset.parameter([-1, -1, LEN], Type.f32, name='qkv')
        if not is_ref:
            mvn = opset.mvn_custom(qkv, axes=[-1], weight=weight, bias=bias, normalize_variance=True, eps=LAYER_NORM_EPS, eps_mode="inside_sqrt", name='mvn')
            if is_i8:
                fq = opset.fake_quantize(mvn, np.array([-127.0], dtype=np.float32), np.array([127.0], dtype=np.float32), np.array([-127.0], dtype=np.float32), np.array([127.0], dtype=np.float32), 255, "NUMPY", name=f'fq_input_0')
                fq_result = opset.convert(fq, np.int8)
            else:
                fq_result = mvn
        else:
            mvn_ref = opset.mvn(qkv, axes=[-1], normalize_variance=True, eps=LAYER_NORM_EPS, eps_mode="inside_sqrt", name='mvn_ref')
            mul = opset.multiply(mvn_ref, weight, auto_broadcast='numpy', name=f'input_layernorm/mul')
            add = opset.add(mul, bias, auto_broadcast='numpy', name=f'input_layernorm/add')
            if is_i8:
                fq_result = opset.fake_quantize(add, np.array([-127.0], dtype=np.float32), np.array([127.0], dtype=np.float32), np.array([-127.0], dtype=np.float32), np.array([127.0], dtype=np.float32), 255, "NUMPY", name=f'fq_input_0')
            else:
                fq_result = add

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
        results.append(np.array(r.data))
    #m = model.get_runtime_model()
    #m = model.net
    #serialize(m, 'ov-mvn.xml', 'ov-mvn.bin')

    return results

def test_mvn():
    qkvs = [
            np.random.random(size=[2, 900, LEN]).astype(np.float32)*10,
            np.random.random(size=[2, 1, LEN]).astype(np.float32)*10,
            ]
    weight = np.random.randint(low=-1, high=2, size=[LEN]).astype(np.float32)
    bias = np.random.randint(low=-1, high=2, size=[LEN]).astype(np.float32)
    qkvs[0][:,:,0::3] = -1
    qkvs[0][:,:,1::3] = +1
    qkvs[0][:,:,2::3] = 0
    qkvs[1][:,:,0::3] = -1
    qkvs[1][:,:,1::3] = +1
    qkvs[1][:,:,2::3] = 0
    
    # print('test ref...')
    # ref_results = gen_ov_i8(qkvs, True)
    # print('\ntest cur...')
    # cur_results = gen_ov_i8(qkvs, False)
    def test_one(is_i8):
        ref_results = gen_ov_mvn_custom(qkvs, weight, bias, is_i8, True)
        cur_results = gen_ov_mvn_custom(qkvs, weight, bias, is_i8, False)
        for (i, ref) in enumerate(ref_results):
            if not np.allclose(ref, cur_results[i], rtol=0.001, atol=0.01):
                print('cur\n', cur_results[i], '\nref\n', ref_results[i])
                print(f'error: idx {i} not close')
        else:
            print('done')
    print('test bf16...')
    test_one(False)
    print('test i8...')
    test_one(True)

test_mvn()
