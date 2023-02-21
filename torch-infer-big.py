import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('model_big')
model = AutoModelForCausalLM.from_pretrained('model_big').to(device).eval()
tokenizer.pad_token = tokenizer.eos_token
df = pd.read_json('results/a100-asparagus-infers.jsonl', lines=True)

f = open('ipex-results-big.org.txt', 'w')
# with torch.cpu.amp.autocast():
#     for j, i in enumerate(df.prompt.iloc[:5]):
#         input_ids = tokenizer.encode(i, return_tensors='pt')
#         if len(input_ids[0]) >= 300:
#             input_ids = input_ids[:, -300:]
#         beg = time.time()
#         outputs = model.generate(input_ids.to(device), pad_token_id=tokenizer.eos_token_id,
#                     num_beams=2, max_new_tokens=100, temperature=1.0)
#         end = time.time()
#         x = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         f.write('\n'.join(x))
#         print(f'{j} cost {end-beg} sec')
#         j += 1
for j, i in enumerate(df.prompt.iloc[:5]):
    input_ids = tokenizer.encode(i, return_tensors='pt')
    if len(input_ids[0]) >= 300:
        input_ids = input_ids[:, -300:]
    beg = time.time()
    outputs = model.generate(input_ids.to(device), pad_token_id=tokenizer.eos_token_id,
                num_beams=2, max_new_tokens=100, temperature=1.0)
    end = time.time()
    x = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    f.write('\n'.join(x))
    print(f'{j} cost {end-beg} sec')
    j += 1
f.close()