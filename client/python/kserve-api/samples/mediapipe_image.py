#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
sys.path.append("../../../../demos/common/python")

import cv2
import numpy as np
import classes
import datetime
import argparse
from client_utils import print_statistics

import tritonclient.grpc as grpcclient


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sends requests via KServe gRPC API using images in format supported by OpenCV. '
                                                 'It displays performance statistics and optionally the model accuracy')
    parser.add_argument('--images_list', required=False, default='input_images.txt', help='path to a file with a list of labeled images')
    parser.add_argument('--grpc_address',required=False, default='localhost',  help='Specify url to grpc service. default:localhost')
    parser.add_argument('--grpc_port',required=False, default=9000, help='Specify port to grpc service. default: 9000')
    parser.add_argument('--input_name',required=False, default='input', help='Specify input tensor name. default: input')
    parser.add_argument('--output_name',required=False, default='resnet_v1_50/predictions/Reshape_1',
                        help='Specify output name. default: resnet_v1_50/predictions/Reshape_1')
    parser.add_argument('--batchsize', default=1,
                        help='Number of images in a single request. default: 1',
                        dest='batchsize')
    parser.add_argument('--model_name', default='resnet', help='Define model name, must be same as is in service. default: resnet',
                        dest='model_name')
    parser.add_argument('--pipeline_name', default='', help='Define pipeline name, must be same as is in service',
                        dest='pipeline_name')
    parser.add_argument('--tls', default=False, action='store_true', help='use TLS communication with GRPC endpoint')

    error = False
    args = vars(parser.parse_args())

    address = "{}:{}".format(args['grpc_address'],args['grpc_port'])
    input_name = args['input_name']
    output_name = args['output_name']

    processing_times = np.zeros((0),int)

    input_images = args.get('images_list')
    with open(input_images) as f:
        lines = f.readlines()
    batch_size = int(args.get('batchsize'))
    while batch_size > len(lines):
        lines += lines

    batch_size = int(args.get('batchsize'))

    print('Start processing:')
    print('\tModel name: {}'.format(args.get('pipeline_name') if bool(args.get('pipeline_name')) else args.get('model_name')))

    iteration = 0
    is_pipeline_request = bool(args.get('pipeline_name'))

    model_name = args.get('pipeline_name') if is_pipeline_request else args.get('model_name')

    try:
        triton_client = grpcclient.InferenceServerClient(
            url=address,
            ssl=args['tls'],
            verbose=False)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    processing_times = np.zeros((0),int)

    total_executed = 0
    matched_count = 0
    batch_i = 0
    image_data = []
    image_binary_size = []
    labels = []
    for line in lines:
        inputs = []
        batch_i += 1
        path, label = line.strip().split(" ")
        data = cv2.imread(path)
        image = np.float32(data)

        inputs.append(grpcclient.InferInput(args['input_name'], [image.shape[0], image.shape[1], image.shape[2]],  "UINT8"))
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput(output_name))

        nmpy = np.array(image.data, np.uint8)
        inputs[0].set_data_from_numpy(nmpy)

        start_time = datetime.datetime.now()
        results = triton_client.infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs)
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        processing_times = np.append(processing_times,np.array([int(duration)]))
        output = results.as_numpy(output_name)
        nu = np.array(output)
        # for object classification models show imagenet class
        print('Iteration {}; Processing time: {:.2f} ms; speed {:.2f} fps'.format(iteration,round(np.average(duration), 2),
                                                                                      round(1000 * batch_size / np.average(duration), 2)
                                                                                      ))
        # Comment out this section for non imagenet datasets
        print("imagenet top results in a single batch:")
        for i in range(nu.shape[0]):
            print("First shape: ", nmpy.shape)
            print("Second shape: ", nu.shape)
            print("Array equal: ", np.array_equal(nmpy, nu))
        if np.array_equal(nmpy, nu):
            matched_count = matched_count + 1
        total_executed = total_executed + 1

    print_statistics(processing_times, batch_size)
    print('Classification accuracy: {:.2f}'.format(100*matched_count/total_executed))
