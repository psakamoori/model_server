#
# Copyright 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import time
import ovmsclient
import torch
import argparse
import blingfire
import numpy as np
#from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description='Demo for GPT-J causal LM requests using ovmsclient gRPC API')

parser.add_argument('--input', required=True, help='Beginning of a sentence', type=str)
parser.add_argument('--url', required=False, help='Url to connect to', type=str, default='localhost:9000')
parser.add_argument('--model_name', required=False, help='Model name in the serving', type=str, default='gpt-j-6b')
parser.add_argument('--eos_token_id', required=False, help='End of sentence token', type=int, default=198)
args = vars(parser.parse_args())

#### blingfire
h = blingfire.load_model(os.path.join(os.path.dirname(blingfire.__file__), "gpt2.bin"))
h_reverse = blingfire.load_model(os.path.join(os.path.dirname(blingfire.__file__), "gpt2.i2w"))
blingfire.change_settings_dummy_prefix(h, False)
blingfire.change_settings_dummy_prefix(h_reverse, False)
# # # # # # use the model from one or more threads
# # # # # print(s)
# # # # # ids = blingfire.text_to_ids(h, s, 128, 100)  # sequence length: 128, oov id: 100
# # # # # print(ids)                                   # returns a numpy array of length 128 (padded or trimmed)

# # # # # print('a:', blingfire.ids_to_text(h_reverse, ids))
#### blingfire

client = ovmsclient.make_grpc_client(args['url'])
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

input_sentence = args['input']
#print(input_sentence, end='', flush=True)

concat_ids_bf = blingfire.text_to_ids(h, input_sentence, 128, 100, no_padding=True).tolist()

iteration = 0
first_latency = -1
last_latency = -1
while True:
    #inputs = tokenizer(input_sentence, return_tensors="np")
    # print(inputs['input_ids'], blingfire.text_to_ids(h, input_sentence, 128, 100, no_padding=True))
    # inputs['input_ids'] = blingfire.text_to_ids(h, input_sentence, 512, no_padding=True).astype('int64').reshape(1, -1)
    #concat_ids_bf = blingfire.text_to_ids(h, input_sentence, 128, 100, no_padding=True).tolist()
    #print(concat_ids_bf)
    #print(tokenizer(input_sentence, return_tensors="np")['input_ids'][0].tolist())
    #tokens = blingfire.text_to_ids(h, input_sentence, 128, 100, no_padding=True)
    #inputs = dict(
    #    input_ids=tokens.astype('int64').reshape(1, -1),
    #    attention_mask=np.ones((1, len(tokens)), dtype='int64'))
    inputs = dict(
        input_ids=np.array(concat_ids_bf, dtype='int64').reshape(1, -1),
        attention_mask=np.ones((1, len(concat_ids_bf)), dtype='int64'))
    start_time = time.time()
    results = client.predict(inputs=dict(inputs), model_name=args['model_name'])
    latency = time.time() - start_time
    if first_latency == -1:
        first_latency = latency
    last_latency = latency
    predicted_token_id = torch.argmax(torch.nn.functional.softmax(torch.Tensor(results[0,-1,:]),dim=-1),dim=-1)
    #word = tokenizer.decode(predicted_token_id)
    #input_sentence += word
    # print(f"Iteration: {iteration}\nLast predicted token: {predicted_token_id}\nLast latency: {last_latency}s\n{input_sentence}")
    #word = blingfire.ids_to_text(h_reverse, np.array([predicted_token_id], dtype='uint32'), skip_special_tokens=False)
    concat_ids_bf.append(predicted_token_id)
    #print(concat_ids_bf)
    #print(np.array(concat_ids_bf).shape)
    #print(f'{word}', end='', flush=True)
    input_sentence = blingfire.ids_to_text(h_reverse, np.array(concat_ids_bf, dtype='uint32'), skip_special_tokens=False)
    #if word[0] == '\'' or word[0] == ',' or word[0] == '.' or word[0] == '-' or input_sentence[-1] == '-' or input_sentence[-1] == '\"':
    #    input_sentence += word
    #else:
    #input_sentence += f'{word}'
    #print(blingfire.ids_to_text(h_reverse, np.array(concat_ids_bf), skip_special_tokens=True))
    iteration += 1
    print(iteration, input_sentence)
    if predicted_token_id == args['eos_token_id']:
        break

# split line below to 3 different lines
print(f"Number of iterations: {iteration}")
print(f"First latency: {first_latency}s")
print(f"Last latency: {last_latency}s")

### blingfire
blingfire.free_model(h_reverse)
blingfire.free_model(h)
### blingfire
