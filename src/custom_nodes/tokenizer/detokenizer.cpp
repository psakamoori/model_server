//*****************************************************************************
// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include "detokenizer.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../../custom_node_interface.h"
#include "../common/utils.hpp"

#include <iostream>
#include <string>
#include <cstring>
#include <chrono>
#include <atomic>
#include <memory>
#include <algorithm>
#include <numeric>

#include "blingfiretokdll.h"

namespace custom_nodes {
namespace detokenizer {

std::atomic<int> maxId{0};

Model::Model(const std::string& modelPath) : id(maxId++) {
    handle = BlingFire::LoadModel(modelPath.c_str());
    std::cout << "[detokenizer] [" << id << "] Model loaded." << std::endl;
}

Model::~Model() {
    if (handle) {
        BlingFire::FreeModel(handle);
        std::cout << "[detokenizer] [" << id << "] Model unloaded." << std::endl;
    }
}

const int Model::detokenize(int32_t* ids, int idsCount, char* buffer, int maxOutUtf8StrByteCount, bool skipSpecialTokens) {
    std::cout << "[detokenizer] [" << id << "] Detokenizing " << idsCount << " tokens..." << std::endl;
    return BlingFire::IdsToText(handle, ids, idsCount, buffer, maxOutUtf8StrByteCount, skipSpecialTokens);
}

std::string Model::detokenizeEx(const std::vector<int64_t>& tokens, int maxOutUtf8StrByteCount) {
    auto ids = std::make_unique<int32_t[]>(tokens.size());
    std::transform(tokens.begin(), tokens.end(), ids.get(),
        [](int64_t val) { return static_cast<int32_t>(val); });
    std::string str(maxOutUtf8StrByteCount + 1, '\0'); // +1 due to null ending
    const int strLength = detokenize(ids.get(), tokens.size(), &str[0], maxOutUtf8StrByteCount, false);
    str.resize(strLength -1);  // remove null terminator
    return std::move(str);
}

}  // namespace detokenizer
}  // namespace custom_nodes

using namespace custom_nodes::detokenizer;

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    std::string modelPath = get_string_parameter("model_path", params, paramsCount, "");
    NODE_ASSERT(!modelPath.empty(), "model_path cannot be empty");
    try {
        *customNodeLibraryInternalManager = new Model(modelPath);
    } catch (...) {
        std::cerr << "[detokenizer] initialize() fail: Cannot load tokenization model from path: " << modelPath << std::endl;
        return 1;
    }
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    if (customNodeLibraryInternalManager != nullptr) {
        Model* manager = static_cast<Model*>(customNodeLibraryInternalManager);
        delete manager;
    }
    return 0;
}

// void softmax(float* input, size_t size) {
// 	int i;
// 	float m, sum, constant;

// 	m = -INFINITY;
// 	for (i = 0; i < size; ++i) {
// 		if (m < input[i]) {
// 			m = input[i];
// 		}
// 	}

// 	sum = 0.0;
// 	for (i = 0; i < size; ++i) {
// 		sum += exp(input[i] - m);
// 	}

// 	constant = m + log(sum);
// 	for (i = 0; i < size; ++i) {
// 		input[i] = exp(input[i] - constant);
// 	}
// }

// in:  [-1, -1, 50400]
// out: [Batch, MaxLength]
int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    std::cout << "[detokenizer] execute()" << std::endl;
    auto start = std::chrono::steady_clock::now();
    // Parameters reading
    int maxBufferLength = get_int_parameter("max_buffer_length", params, paramsCount, -1);
    NODE_ASSERT(maxBufferLength > 0, "max_buffer_length param must be larger than 0");
    
    // Inputs reading
    const CustomNodeTensor* logitsTensor = nullptr;

    for (int i = 0; i < inputsCount; i++) {
        if (std::strcmp(inputs[i].name, "logits") == 0) {
            logitsTensor = &(inputs[i]);
        } else {
            std::cerr << "Unrecognized input: " << inputs[i].name << std::endl;
            return 1;
        }
    }

    // Validating inputs
    NODE_ASSERT(logitsTensor != nullptr, "Missing logits input");
    NODE_ASSERT(logitsTensor->precision == FP32, "logits input is not FP32");

    NODE_ASSERT(logitsTensor->dimsCount == 3, "input logits shape must have 3 dimensions");
    NODE_ASSERT(logitsTensor->dims[0] > 0, "input text dimension 1 must be larger than 0");
    NODE_ASSERT(logitsTensor->dims[1] > 0, "input text dimension 2 must be larger than 0");
    NODE_ASSERT(logitsTensor->dims[2] > 0, "input text dimension 3 must be larger than 0");

    Model* model = static_cast<Model*>(customNodeLibraryInternalManager);

    std::vector<std::string> results;
    for (uint64_t batch = 0; batch < logitsTensor->dims[0]; batch++) {
        std::cout << "[detokenizer] slicing batch " << batch << std::endl;
        // slice
        float* logits = reinterpret_cast<float*>(
            logitsTensor->data + 
                batch * (logitsTensor->dims[1] * logitsTensor->dims[2] * sizeof(float)) +   // offset by batch
                ((logitsTensor->dims[1] - 1) * logitsTensor->dims[2] * sizeof(float)));     // offset to get last element of second dimension

        // argmax
        std::cout << "[detokenizer] argmax batch " << batch << std::endl;
        float* result = std::max_element(logits, logits + logitsTensor->dims[2]);
        int32_t token = std::distance(logits, result);

        // detokenize
        std::cout << "[detokenizer] (token " << token << ") detokenize batch " << batch << std::endl;
        std::vector<int64_t> tokens = {token};
        auto text = model->detokenizeEx(tokens, maxBufferLength);
        results.push_back(text);
        std::cout << "[detokenizer] text: " << text << std::endl;
    }

    std::cout << "[detokenizer] getting max string length" << std::endl;
    size_t maxStringLength = 0;
    for (const auto& str : results) {
        maxStringLength = std::max(maxStringLength, str.size());
    }
    size_t width = maxStringLength + 1;

    std::cout << "[detokenizer] prepraing output" << std::endl;
    *outputsCount = 1;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));
    if ((*outputs) == nullptr) {
        std::cerr << "malloc has failed" << std::endl;
        return 1;
    }

    // Outputs allocation
    CustomNodeTensor& output = (*outputs)[0];
    output.name = "texts";
    output.dataBytes = width * results.size();
    output.data = (uint8_t*)malloc(output.dataBytes);
    output.dimsCount = 2;
    output.dims = (uint64_t*)malloc(output.dimsCount * sizeof(uint64_t));
    NODE_ASSERT(output.dims != nullptr, "malloc has failed");
    output.dims[0] = results.size();
    output.dims[1] = width;
    output.precision = C_STRING_ARRAY;

    std::cout << "[detokenizer] writing output" << std::endl;
    for (size_t i = 0; i < results.size(); i++) {
        std::memcpy(output.data + i * width, results[i].data(), results[i].size());
        output.data[i * width + results[i].size()] = 0;
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "[detokenizer] Elapsed time in seconds: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
         << " ms" << std::endl;
    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = "logits";
    (*info)[0].dimsCount = 3;
    (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = -1;
    (*info)[0].dims[1] = -1;
    (*info)[0].dims[2] = -1;
    (*info)[0].precision = FP32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = "texts";
    (*info)[0].dimsCount = 2;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = -1;
    (*info)[0].dims[1] = -1;
    (*info)[0].precision = C_STRING_ARRAY;

    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    free(ptr);
    return 0;
}
