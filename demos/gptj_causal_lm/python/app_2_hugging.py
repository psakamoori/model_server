import os
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

# input
s = "This is a test. Эpple pie. How do I renew my virtual smart card? ends"
print(s)

inputs = tokenizer(s, return_tensors="np")
ids = inputs['input_ids']
print(ids)

text = tokenizer.decode(ids[0])
print(text)
