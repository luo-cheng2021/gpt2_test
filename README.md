# gpt2_test

Test scripts for gpt2 based models.
Based on https://github.com/LowinLi/fastgpt.

## prepare ov package
source ~/intel/oneapi/setvars.sh
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
    -DENABLE_TESTS=ON \
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

## prepare python package
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

## parepare model
copy model+results to current directory
python /path/to/ov_src/tools/gpt/gpt_neox_ov.py gpt_neox_ov.py /path/to/pytorch/model /path/to/ov/IR

## run ipex
python torch-infer-ipex.py

## run ov
source ~/intel/oneapi/setvars.sh
source path/to/ov/setupvars.sh

python torch-infer-ov2-big_attn_dyn.py model_big /path/to/ov/IR

## compare accuracy
cmp ipex-results.txt ov-results.txt # should show nothing
