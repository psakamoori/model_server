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

#include <stdlib.h>
#include <stdio.h>

#include "../../custom_node_interface.h"
#include "../common/utils.hpp"

#include <iostream>
#include <string>
#include <cstring>
#include <chrono>

#include "blingfiretokdll.h"

int main();

class TokenizerModel {
    void* handle = nullptr;

public:
    TokenizerModel(const std::string& modelPath) {
        handle = BlingFire::LoadModel(modelPath.c_str());
        std::cout << "MMM Model loaded." << std::endl;
    }

    ~TokenizerModel() {
        if (handle) {
            BlingFire::FreeModel(handle);
            std::cout << "MMM Model unloaded." << std::endl;
        }
    }

    const int tokenize(const std::string& text, int32_t* ids, int maxIdsArrLength) {
        std::cout << "MMM Tokenizing: [" << text << "]" << std::endl;
        return BlingFire::TextToIds(this->handle, text.c_str(), text.size(), ids, maxIdsArrLength);
    }
};

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    std::string modelPath = get_string_parameter("model_path", params, paramsCount, "");
    NODE_ASSERT(!modelPath.empty(), "model_path cannot be empty");
    auto* manager = new TokenizerModel(modelPath);
    *customNodeLibraryInternalManager = manager;
    std::cout << "MMM Initializing tokenizer with path " << modelPath << ", "  << (uint64_t)*customNodeLibraryInternalManager % 511 << std::endl;
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    if (customNodeLibraryInternalManager != nullptr) {
        std::cout << "MMM Deinitializing tokenizer: " << (uint64_t)customNodeLibraryInternalManager % 511 << std::endl;
        TokenizerModel* manager = static_cast<TokenizerModel*>(customNodeLibraryInternalManager);
        delete manager;
    }
    return 0;
}

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    auto start = std::chrono::steady_clock::now();
    std::cout << "MMM Using tokenizer: " << (uint64_t)customNodeLibraryInternalManager % 511 << std::endl;
    // Parameters reading
    int maxIdsArrLength = get_int_parameter("max_ids_arr_length", params, paramsCount, -1);
    NODE_ASSERT(maxIdsArrLength > 0, "max ids array length must be larger than 0");

    // Inputs reading
    const CustomNodeTensor* textTensor = nullptr;

    for (int i = 0; i < inputsCount; i++) {
        if (std::strcmp(inputs[i].name, "text") == 0) {
            textTensor = &(inputs[i]);
        } else {
            std::cout << "Unrecognized input: " << inputs[i].name << std::endl;
            return 1;
        }
    }

    // Validating inputs
    NODE_ASSERT(textTensor != nullptr, "Missing text input");
    NODE_ASSERT(textTensor->precision == U8, "text input is not U8");

    NODE_ASSERT(textTensor->dimsCount == 2, "input text shape must have 2 dimensions");
    NODE_ASSERT(textTensor->dims[0] == 1, "input text dimension 1 must be batch 1 for now");
    NODE_ASSERT(textTensor->dims[1] > 0, "input text dimension 2 must be larger than 0");

    std::cout << "maxIdsArrLength: [" << maxIdsArrLength << "]" << std::endl;

    // Convert textTensor to std::string
    std::string text((const char*)textTensor->data, textTensor->dataBytes);
    std::cout << "Received input: [" << text << "]" << std::endl;

    TokenizerModel* manager = static_cast<TokenizerModel*>(customNodeLibraryInternalManager);
    
    int32_t* ids = (int32_t*)malloc(maxIdsArrLength * sizeof(int32_t));
    auto idsCount = manager->tokenize(text, ids, maxIdsArrLength);

    // TODO: Assert for idsCount <= maxIdsArrLength, free previously allocated memory

    // Convert ids to int64_t dynamically allocated array
    int64_t* ids_i64 = (int64_t*)malloc(idsCount * sizeof(int64_t));
    int64_t* attention_i64 = (int64_t*)malloc(idsCount * sizeof(int64_t));
    for (int i = 0; i < idsCount; i++) {
        ids_i64[i] = static_cast<int64_t>(ids[i]);
        attention_i64[i] = 1;
    }
    free(ids);

    // Write output
    *outputsCount = 2;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));
    if ((*outputs) == nullptr) {
        std::cerr << "malloc has failed" << std::endl;
        free(ids);
        return 1;
    }

    CustomNodeTensor& output = (*outputs)[0];
    output.name = "tokens";
    output.data = reinterpret_cast<uint8_t*>(ids_i64);
    output.dataBytes = sizeof(int64_t) * idsCount;
    output.dimsCount = 2;
    output.dims = (uint64_t*)malloc(output.dimsCount * sizeof(uint64_t));
    NODE_ASSERT(output.dims != nullptr, "malloc has failed");
    output.dims[0] = 1;
    output.dims[1] = idsCount;
    output.precision = I64;

    CustomNodeTensor& attention = (*outputs)[1];
    attention.name = "attention";
    attention.data = reinterpret_cast<uint8_t*>(attention_i64);
    attention.dataBytes = sizeof(int64_t) * idsCount;
    attention.dimsCount = 2;
    attention.dims = (uint64_t*)malloc(attention.dimsCount * sizeof(uint64_t));
    NODE_ASSERT(attention.dims != nullptr, "malloc has failed");
    attention.dims[0] = 1;
    attention.dims[1] = idsCount;
    attention.precision = I64;

    auto end = std::chrono::steady_clock::now();
    std::cout << "MMM Elapsed time in seconds: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
         << " ms" << std::endl;

    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = "text";
    (*info)[0].dimsCount = 2;
    (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = -1;
    (*info)[0].dims[1] = -1;
    (*info)[0].precision = U8;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 2;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = "tokens";
    (*info)[0].dimsCount = 2;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = -1;
    (*info)[0].dims[0] = -1;
    (*info)[0].precision = I64;

    (*info)[1].name = "attention";
    (*info)[1].dimsCount = 2;
    (*info)[1].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[1].dims) != nullptr, "malloc has failed");
    (*info)[1].dims[0] = -1;
    (*info)[1].dims[0] = -1;
    (*info)[1].precision = I64;
    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    free(ptr);
    return 0;
}
