import os
import blingfire
import numpy as np

# one time load the model (we are using the one that comes with the package)
h = blingfire.load_model(os.path.join(os.path.dirname(blingfire.__file__), "gpt2.bin"))
h_i2w = blingfire.load_model(os.path.join(os.path.dirname(blingfire.__file__), "gpt2.i2w"))

# input
s = "This is a test. Эpple pie. How do I renew my virtual smart card? ends"
print(s)

ids = blingfire.text_to_ids(h, s, 128, no_padding=True)  # sequence length: 128, oov id: 100
print(ids)                                   # returns a numpy array of length 128 (padded or trimmed)

print(np.array(
    [  770,   318,   257,  1332,    13, 12466,   255,   381,   293,  2508,    13,  1374,
   466,   314,  6931,   616,  7166,  4451,  2657,    30,  5645], dtype=np.uint32))
text = blingfire.ids_to_text(h_i2w, np.array(
    [  770,   318,   257,  1332,    13, 12466,   255,   381,   293,  2508,    13,  1374,
   466,   314,  6931,   616,  7166,  4451,  2657,    30,  5645], dtype=np.uint32), skip_special_tokens=False)     # generate text with special tokens included
print(text)                                  


# free the model at the end
blingfire.free_model(h)
blingfire.free_model(h_i2w)
