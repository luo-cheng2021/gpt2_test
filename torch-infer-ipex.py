import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('model')
model = AutoModelForCausalLM.from_pretrained('model').to(device).eval()
org_forward = model.transformer.forward
stat = {
    'init': 0,
    'infer_1x300': 0,
    'infer_1x1': 0,
    'post': 0,
    'times': 0
    }
def myforward(self, *args, **kwargs):
    beg = time.time()
    ret = org_forward(self, *args, **kwargs)
    end = time.time()
    if stat['times'] == 0:
        stat['infer_1x300'] += end - beg
    else:
        stat['infer_1x1'] += end - beg
    stat['times'] += 1
    return ret
model.transformer.forward = myforward

tokenizer.pad_token = tokenizer.eos_token
df = pd.read_json('results/a100-asparagus-infers.jsonl', lines=True)

import intel_extension_for_pytorch as ipex
model = model.to(memory_format=torch.channels_last)
model = ipex.optimize(model, dtype=torch.bfloat16, level="O1")
#print(model)
use_profile = False
f = open('ipex-results.txt', 'w')
with torch.cpu.amp.autocast():
    for j, i in enumerate(df.prompt.iloc[:5]):
        input_ids = tokenizer.encode(i, return_tensors='pt')
        if len(input_ids[0]) >= 300:
            input_ids = input_ids[:, -300:]
        beg = time.time()
        if use_profile:
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof, record_function("model_inference"):
                outputs = model.generate(input_ids.to(device), pad_token_id=tokenizer.eos_token_id,
                            num_beams=2, max_new_tokens=100, temperature=1.0)
        else:
            outputs = model.generate(input_ids.to(device), pad_token_id=tokenizer.eos_token_id,
                            num_beams=2, max_new_tokens=100, temperature=1.0)
        end = time.time()
        x = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        f.write('\n'.join(x))
        f.write(f'\n{j} ==============================\n')
        print(f'{j} cost {end-beg:.2f} sec, stat: {stat}')
        stat = {
            'init': 0,
            'infer_1x300': 0,
            'infer_1x1': 0,
            'post': 0,
            'times': 0
        }

f.close()
if use_profile:
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=-1))
