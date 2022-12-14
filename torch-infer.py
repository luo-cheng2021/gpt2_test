import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('model')
model = AutoModelForCausalLM.from_pretrained('model').to(device).eval()
tokenizer.pad_token = tokenizer.eos_token
df = pd.read_json('results/a100-asparagus-infers.jsonl', lines=True)
j = 0

import intel_extension_for_pytorch as ipex
model = ipex.optimize(model)

for i in df.prompt.iloc[:5]:
    input_ids = tokenizer.encode(i, return_tensors='pt')
    if len(input_ids[0]) >= 300:
        input_ids = input_ids[:, -300:]
    beg = time.time()
    outputs = model.generate(input_ids.to(device), pad_token_id=tokenizer.eos_token_id,
                   num_beams=2, max_new_tokens=100, temperature=1.0)
    end = time.time()
    x = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f'{j} cost {end-beg} sec: {x}')
    j += 1

