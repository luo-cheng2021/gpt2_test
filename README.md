# gpt2_test

Test scripts for gpt2 based models.
Based on https://github.com/LowinLi/fastgpt.

## prepare ov package
cmake -DENABLE_OV_ONNX_FRONTEND=ON ...

## prepare python package
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

## parepare model
copy model+results to current directory

## run ipex
python torch-infer-ipex.py

## run ov
source path/to/ov/setupvars.sh

python torch-infer-ov.py

## compare accuracy
cmp ipex-results.txt ov-results.txt # should show nothing
