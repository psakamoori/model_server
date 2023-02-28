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

#include "../../../custom_node_interface.h"
#include "../../common/utils.hpp"

#include <iostream>
#include <string>
#include <cstring>
#include <chrono>
#include <atomic>
#include <memory>
#include <algorithm>
#include <numeric>

#include "model.hpp"

using namespace custom_nodes::tokenizer;

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    std::string modelPath = get_string_parameter("model_path", params, paramsCount, "");
    NODE_ASSERT(!modelPath.empty(), "model_path cannot be empty");
    try {
        *customNodeLibraryInternalManager = new BlingFireModel(modelPath);
    } catch (...) {
        std::cerr << "[tokenizer] initialize() fail: Cannot load tokenization model from path: " << modelPath << std::endl;
        return 1;
    }
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    if (customNodeLibraryInternalManager != nullptr) {
        BlingFireModel* manager = static_cast<BlingFireModel*>(customNodeLibraryInternalManager);
        delete manager;
    }
    return 0;
}

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    auto start = std::chrono::steady_clock::now();
    // Parameters reading
    int maxIdsArrLength = get_int_parameter("max_ids_arr_length", params, paramsCount, -1);
    NODE_ASSERT(maxIdsArrLength > 0, "max_ids_arr_length param must be larger than 0");

    // Inputs reading
    const CustomNodeTensor* textTensor = nullptr;

    for (int i = 0; i < inputsCount; i++) {
        if (std::strcmp(inputs[i].name, "texts") == 0) {
            textTensor = &(inputs[i]);
        } else {
            std::cerr << "Unrecognized input: " << inputs[i].name << std::endl;
            return 1;
        }
    }

    // Validating inputs
    NODE_ASSERT(textTensor != nullptr, "Missing text input");
    NODE_ASSERT(textTensor->precision == U8, "text input is not U8");

    NODE_ASSERT(textTensor->dimsCount == 2, "input text shape must have 2 dimensions");
    NODE_ASSERT(textTensor->dims[0] > 0, "input text dimension 1 must be larger than 0 (number of texts)");
    NODE_ASSERT(textTensor->dims[1] > 0, "input text dimension 2 must be larger than 0 (max null terminated text length)");

    BlingFireModel* model = static_cast<BlingFireModel*>(customNodeLibraryInternalManager);

    *outputsCount = 2;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));
    if ((*outputs) == nullptr) {
        std::cerr << "malloc has failed" << std::endl;
        return 1;
    }

    std::vector<std::vector<int64_t>> ids(textTensor->dims[0]);
    // For each batch, sequentially
    for (uint64_t i = 0; i < textTensor->dims[0]; i++) {
        const char* strStart = (const char*)textTensor->data + i * textTensor->dims[1];
        std::string text(strStart, std::strlen(strStart));  // We are ensure this is 0 terminated by the server
        ids[i] = model->tokenize(text, maxIdsArrLength);
    }

    size_t maxTokenSize = 0;
    for (const auto& id : ids) {
        maxTokenSize = std::max(maxTokenSize, id.size());
    }

    CustomNodeTensor& output = (*outputs)[0];
    output.name = "tokens";
    output.dataBytes = sizeof(int64_t) * maxTokenSize * ids.size();
    output.data = (uint8_t*)malloc(output.dataBytes);
    output.dimsCount = 2;
    output.dims = (uint64_t*)malloc(output.dimsCount * sizeof(uint64_t));
    NODE_ASSERT(output.dims != nullptr, "malloc has failed");
    output.dims[0] = ids.size();
    output.dims[1] = maxTokenSize;
    output.precision = I64;

    CustomNodeTensor& attention = (*outputs)[1];
    attention.name = "attention";
    attention.dataBytes = sizeof(int64_t) * maxTokenSize * ids.size();
    attention.data = (uint8_t*)malloc(attention.dataBytes);
    attention.dimsCount = 2;
    attention.dims = (uint64_t*)malloc(attention.dimsCount * sizeof(uint64_t));
    NODE_ASSERT(attention.dims != nullptr, "malloc has failed");
    attention.dims[0] = ids.size();
    attention.dims[1] = maxTokenSize;
    attention.precision = I64;

    std::cout << "[tokenizer] tokens.dim[0]==" << output.dims[0] << std::endl;
    std::cout << "[tokenizer] tokens.dim[1]==" << output.dims[1] << std::endl;

    for (size_t i = 0; i < ids.size(); i++) {
        std::memcpy(output.data + i * maxTokenSize * sizeof(int64_t), ids[i].data(), ids[i].size() * sizeof(int64_t));
        for (size_t j = 0; j < ids[i].size(); j++) {
            ((int64_t*)attention.data)[i * maxTokenSize + j] = 1;
        }
        for (size_t j = ids[i].size(); j < maxTokenSize; j++) {
            ((int64_t*)attention.data)[i * maxTokenSize + j] = 0;
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "[tokenizer] Elapsed time in seconds: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
         << " ms" << std::endl;

    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = "texts";
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
    (*info)[0].dims[1] = -1;
    (*info)[0].precision = I64;

    (*info)[1].name = "attention";
    (*info)[1].dimsCount = 2;
    (*info)[1].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[1].dims) != nullptr, "malloc has failed");
    (*info)[1].dims[0] = -1;
    (*info)[1].dims[1] = -1;
    (*info)[1].precision = I64;
    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    free(ptr);
    return 0;
}
