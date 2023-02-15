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
    std::cout << "MMM Using tokenizer: " << (uint64_t)customNodeLibraryInternalManager % 511 << std::endl;
    // Parameters reading
    // std::string modelPath = get_string_parameter("model_path", params, paramsCount, "unknown");
    int maxIdsArrLength = get_int_parameter("max_ids_arr_length", params, paramsCount, -1);
    // NODE_ASSERT(modelPath != "unknown", "model path is required");
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

    NODE_ASSERT(textTensor->dimsCount == 1, "input text shape must have 1 dimension");
    NODE_ASSERT(textTensor->dims[0] > 0, "input text dimension must be larger than 0");

    std::cout << "maxIdsArrLength: [" << maxIdsArrLength << "]" << std::endl;

    // Convert textTensor to std::string
    std::string text((const char*)textTensor->data, textTensor->dataBytes);
    std::cout << "Received input: [" << text << "]" << std::endl;

    TokenizerModel* manager = static_cast<TokenizerModel*>(customNodeLibraryInternalManager);
    
    int32_t* ids = (int32_t*)malloc(maxIdsArrLength * sizeof(int32_t));
    auto idsCount = manager->tokenize(text, ids, maxIdsArrLength);

    // TODO: Assert for idsCount <= maxIdsArrLength

    // Write output
    *outputsCount = 1;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));
    if ((*outputs) == nullptr) {
        std::cerr << "malloc has failed" << std::endl;
        free(ids);
        return 1;
    }

    CustomNodeTensor& output = (*outputs)[0];
    output.name = "tokens";
    output.data = reinterpret_cast<uint8_t*>(ids);
    output.dataBytes = sizeof(int32_t) * idsCount;
    output.dimsCount = 2;
    output.dims = (uint64_t*)malloc(output.dimsCount * sizeof(uint64_t));
    NODE_ASSERT(output.dims != nullptr, "malloc has failed");
    output.dims[0] = 1;
    output.dims[1] = idsCount;
    output.precision = I32;

    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = "text";
    (*info)[0].dimsCount = 1;
    (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = -1;
    (*info)[0].precision = U8;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));
    NODE_ASSERT((*info) != nullptr, "malloc has failed");

    (*info)[0].name = "tokens";
    (*info)[0].dimsCount = 2;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    NODE_ASSERT(((*info)[0].dims) != nullptr, "malloc has failed");
    (*info)[0].dims[0] = -1;
    (*info)[0].dims[0] = -1;
    (*info)[0].precision = I64;
    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    free(ptr);
    return 0;
}

int main() {
    const std::string model_bin = "/ovms/src/custom_nodes/tokenizer/gpt2.bin";
    const std::string model_i2w = "/ovms/src/custom_nodes/tokenizer/gpt2.i2w";

    const std::string input_sentence = "Like Curiosity, the Perseverance rover was built by engineers and scientists at NASA's Jet Propulsion Laboratory in Pasadena, California. Roughly 85% of Perseverance's mass is based on Curiosity \"heritage hardware,\" saving NASA time and money and reducing risk considerably, agency officials have said.  Как и Curiosity, марсоход Perseverance был построен инженерами и учеными из Лаборатории реактивного движения НАСА в Пасадене, Калифорния. По словам официальных лиц агентства, примерно 85% массы Perseverance основано на «традиционном оборудовании» Curiosity, что экономит время и деньги NASA и значительно снижает риски.";

    std::cout << "Loading the model..." << std::endl;
    void* h = BlingFire::LoadModel(model_bin.c_str());
    void* h_reverse = BlingFire::LoadModel(model_i2w.c_str());
    std::cout << "Loaded." << std::endl;

    int32_t ids[1024]={0,};
    int32_t expected_ids[1024] = {770, 318, 257, 1332, 13, 12466, 255, 381, 293, 2508, 13, 1374,
        466, 314, 6931, 616, 7166, 4451, 2657, 30, 5645};
    int num_of_ids = BlingFire::TextToIds(h, input_sentence.c_str(), input_sentence.size(), ids, 1024);

    // std::cout << "actual:  ";
    // for (int i = 0; i < 40; i++) {
    //     std::cout << ids[i] << " ";
    // }
    // std::cout << std::endl << "expected:";
    // for (int i = 0; i < 40; i++) {
    //     std::cout << expected_ids[i] << " ";
    // }
    // std::cout << std::endl;

    char output_buffer[1024];
    BlingFire::IdsToText(h_reverse, ids, num_of_ids, output_buffer, 1024, false);

    std::cout << "Num of ids: " << num_of_ids << std::endl;
    std::cout << "Input: \t\t " << input_sentence << std::endl;
    std::cout << "Conversion back: " << output_buffer << std::endl;

    std::cout << "Freeing the model..." << std::endl;
    BlingFire::FreeModel(h_reverse);
    BlingFire::FreeModel(h);
    std::cout << "Freed." << std::endl;
    return 0;
}
