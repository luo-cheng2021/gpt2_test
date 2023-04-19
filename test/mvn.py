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

def gen_ov_i8(qkvs, high):
    def make_model():
        # [batch, tokens, 2560]
        qkv = opset.parameter([-1, -1, 2560], Type.f32, name='qkv')

        mvn = opset.mvn(qkv, axes=[-1], normalize_variance=True, eps=LAYER_NORM_EPS, eps_mode="inside_sqrt", name='mvn')
        fq = opset.fake_quantize(mvn, -127, 127, -127, 127, 255, "NUMPY", name=f'fq_input_0')
        tmp = opset.convert(fq, np.int8)
        mvn_ref = opset.mvn(qkv, axes=[-1], normalize_variance=True, eps=LAYER_NORM_EPS, eps_mode="inside_sqrt", name='mvn_ref')
        fq_ref = opset.fake_quantize(mvn_ref, -127, 127, -127, 127, 255, "NUMPY", name=f'fq_input_0')
        
        return Model([tmp, fq_ref], [qkv])
    net = make_model()
    core = Core()
    config = {'PERFORMANCE_HINT': '',
            'NUM_STREAMS': '1',
            'ENFORCE_BF16': 'YES'
            }
    model = core.compile_model(net, 'CPU', config)
    req = model.create_infer_request()
    results = []
    results_f32 = []
    for (i, qkv) in enumerate(qkvs):
        inputs = {
            0: Tensor(qkv),
        }
        req.set_input_tensors(inputs)
        req.infer()
        r, r_f32 = req.outputs
        results.append(np.array(r.data))
        results_f32.append(np.array(r_f32.data))
    #m = model.get_runtime_model()
    #m = model.net
    #serialize(m, 'ov-mvn.xml', 'ov-mvn.bin')

    return results, results_f32

def test_mvn():
    qkvs = [
            np.random.random(size=[2, 900, 2560]).astype(np.float32)*10,
            np.random.random(size=[2, 1, 2560]).astype(np.float32)*10,
            ]
    qkvs[0][:,:,0::3] = -1
    qkvs[0][:,:,1::3] = +1
    qkvs[0][:,:,2::3] = 0

    i8_results, f32_results = gen_ov_i8(qkvs, max)
    for (i, ref) in enumerate(f32_results):
        if not np.allclose(ref, i8_results[i], rtol=0.001, atol=0.01):
            print(ref, '\n', i8_results[i], '\n', f32_results[i])
            print(f'error: idx {i} not close')
    else:
        print('done')

test_mvn()
