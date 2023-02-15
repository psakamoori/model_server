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

#include <iostream>
#include <string>

#include "blingfiretokdll.h"

int main();

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    main();
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager) {
    return 0;
}

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    return 1;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    return 1;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    return 1;
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
