import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PaddingStrategy
import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('model_big')
model = AutoModelForCausalLM.from_pretrained('model_big').to(device).eval()
# tokenizer = AutoTokenizer.from_pretrained("CarperAI/FIM-NeoX-1.3B")
# model = AutoModelForCausalLM.from_pretrained("CarperAI/FIM-NeoX-1.3B")

#tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
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
use_pad = True
for j, i in enumerate(df.prompt.iloc[:5]):
    if use_pad:
        input_ids = tokenizer.encode(i, padding=PaddingStrategy.MAX_LENGTH, max_length=900, return_tensors='pt', add_special_tokens=False)
    else:
        input_ids = tokenizer.encode(i, return_tensors='pt', add_special_tokens=False)
    if len(input_ids[0]) >= 900:
        input_ids = input_ids[:, -900:]
    beg = time.time()
    outputs = model.generate(input_ids.to(device), pad_token_id=tokenizer.pad_token_id,
                num_beams=2, max_new_tokens=90, temperature=1.0)
    end = time.time()
    x = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    f.write('\n'.join(x))
    print(f'{j} cost {end-beg} sec')
    j += 1
f.close()