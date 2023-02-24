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
import grpc
import time
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow import make_tensor_proto, make_ndarray
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

parser = argparse.ArgumentParser(description='Demo for GPT-J causal LM requests using ovmsclient gRPC API')

parser.add_argument('--input', required=True, help='Beginning of a sentence', type=str)
parser.add_argument('--url', required=False, help='Url to connect to', type=str, default='localhost:9000')
parser.add_argument('--model_name', required=False, help='Model name in the serving', type=str, default='gpt-j-6b')
parser.add_argument('--eos_token_id', required=False, help='End of sentence token', type=int, default=198)
args = vars(parser.parse_args())

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

channel = grpc.insecure_channel(args['url'],options=[
        ('grpc.max_send_message_length', 1024*1024*1024),
        ('grpc.max_receive_message_length', 1024*1024*1024),
    ])
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

input_sentence = args['input']
print(input_sentence, end='', flush=True)

iteration = 0
first_latency = -1
last_latency = -1
while True:
    start_time = time.time()
    request = predict_pb2.PredictRequest()
    request.model_spec.name = args['model_name']
    request.inputs['texts'].CopyFrom(make_tensor_proto([input_sentence]))
    results = stub.Predict(request, 10.0)
    results = make_ndarray(results.outputs['logits'])
    latency = time.time() - start_time
    if first_latency == -1:
        first_latency = latency
    last_latency = latency
    predicted_token_id = token = torch.argmax(torch.nn.functional.softmax(torch.Tensor(results[0,-1,:]),dim=-1),dim=-1)
    word = tokenizer.decode(predicted_token_id)
    input_sentence += word
    # print(f"Iteration: {iteration}\nLast predicted token: {predicted_token_id}\nLast latency: {last_latency}s\n{input_sentence}")
    print(word, end='', flush=True)
    iteration += 1
    if predicted_token_id == args['eos_token_id']:
        break

# split line below to 3 different lines
print(f"Number of iterations: {iteration}")
print(f"First latency: {first_latency}s")
print(f"Last latency: {last_latency}s")
