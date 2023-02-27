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


// in:  [-1, -1, 50400]
// out: [Batch, MaxLength]
int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
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


    for (uint64_t batch = 0; batch < logitsTensor->dims[0]; i++) {
        // slice
        
        // softmax

        // argmax

        // detokenize
    }

    // Write output

    auto end = std::chrono::steady_clock::now();
    std::cout << "[detokenizer] Elapsed time in seconds: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
         << " ms" << std::endl;
    return 1;
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
