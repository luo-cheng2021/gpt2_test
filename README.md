# gpt2_test

Test scripts for gpt2 based models.
Based on https://github.com/LowinLi/fastgpt.

## [opt] python venv
```
python3.8 -m venv .env
source .env/bin/activate
```

## prepare specific ov package
```
git clone https://github.com/luo-cheng2021/openvino.git -b luocheng/gpt_neox
cd openvino && git submodule update --init --recursive
source ~/intel/oneapi/setvars.sh
mkdir build && cd build
cmake .. -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc \
    -DENABLE_CPU_DEBUG_CAPS=ON \
    -DENABLE_DEBUG_CAPS=ON  \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_INTEL_MYRIAD_COMMON=OFF \
    -DENABLE_INTEL_GNA=OFF \
    -DENABLE_OPENCV=OFF \
    -DENABLE_CPPLINT=OFF \
    -DENABLE_CPPLINT_REPORT=OFF \
    -DENABLE_NCC_STYLE=OFF \
    -DENABLE_TESTS=OFF \
    -DENABLE_OV_CORE_UNIT_TESTS=OFF \
    -DENABLE_INTEL_CPU=ON \
    -DENABLE_INTEL_GPU=OFF \
    -DENABLE_AUTO=OFF \
    -DENABLE_AUTO_BATCH=OFF \
    -DENABLE_MULTI=OFF \
    -DENABLE_HETERO=OFF \
    -DENABLE_INTEL_GNA=OFF \
    -DENABLE_PROFILING_ITT=ON\
    -DENABLE_SAMPLES=ON \
    -DENABLE_PYTHON=ON \
    -DENABLE_TEMPLATE=OFF  \
    -DENABLE_OV_ONNX_FRONTEND=ON \
    -DENABLE_OV_PADDLE_FRONTEND=OFF \
    -DENABLE_OV_PYTORCH_FRONTEND=OFF \
    -DENABLE_OV_TF_FRONTEND=OFF \
    -DENABLE_OPENVINO_DEBUG=OFF \
    -DENABLE_CPU_DEBUG_CAPS=ON \
    -DCMAKE_INSTALL_PREFIX=`pwd`/install \
    -DCMAKE_INSTALL_RPATH=`pwd`/install/runtime/3rdparty/tbb/lib:`pwd`/install/runtime/3rdparty/hddl/lib:`pwd`/install/runtime/lib/intel64 \
    -Dgflags_Dir=`pwd`/../thirdparty/gflags/gflags/cmake
make -j 64 install
source install/setupvars.sh
cd ../../
```

## prepare python package
```
git clone https://github.com/luo-cheng2021/gpt2_test.git
cd gpt2_test
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install -r requirements.txt
```

## prepare bf16 model
```
# make link to model_big or just copy them
# ln -s ~/luocheng/gpt2_test/model_big ./model_big
# ln -s ~/luocheng/gpt2_test/results ./results
cp /path/model+results ./model_big
python ../openvino/tools/gpt/gpt_neox_ov.py ./model_big /path/to/ov/IR
```

## [opt]prepare int8 model
```
# make link to model_big or just copy them
# ln -s ~/luocheng/gpt2_test/model_big ./model_big
# ln -s ~/luocheng/gpt2_test/results ./results
cp /path/model+results ./model_big
python ../openvino/tools/gpt/gpt_neox_ov.py ./model_big /path/to/ov/INT8-IR /path/to/ov/quantized-INT8-IR
```

## run torch with bf16
```
numactl --localalloc -C0-55 python torch-infer-ipex-big.py
```

## run just compiled ov with bf16
```
# if skip prepare stage may need to exec the following 2 lines
# source ~/intel/oneapi/setvars.sh
# source path/to/ov/install/setupvars.sh

OMP_NUM_THREADS=1 numactl --localalloc -C0-55 python torch-infer-ov2-big_attn_dyn.py model_big /path/to/ov/IR
```

## [opt]run just compiled ov with int8
```
# if skip prepare stage may need to exec the following 2 lines
# source ~/intel/oneapi/setvars.sh
# source path/to/ov/install/setupvars.sh

OMP_NUM_THREADS=1 numactl --localalloc -C0-55 python torch-infer-ov2-big_attn_dyn.py model_big /path/to/ov/INT8-IR
```

## run master ov
```
# source path/to/ov/install/setupvars.sh
OMP_NUM_THREADS=1 numactl --localalloc -C0-55 python torch-infer-ov.py
```
