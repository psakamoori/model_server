//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "custom_node_interface.h"  // NOLINT
#include "clip_model.hpp"

#define TEST_VOCAB_FILE_PATH "./vocab.json"
#define TEST_MERGES_FILE_PATH "./merges.txt"

#define INPUT_NAME_TEXTS "texts"

#define OUTPUT_NAME_TOKENS "input_ids"
#define OUTPUT_NAME_ATTENTION "attention_mask"

using namespace custom_nodes::tokenizer;

TEST(ClipTokenizerTest, Run) {
    ClipModel model(TEST_VOCAB_FILE_PATH, TEST_MERGES_FILE_PATH);
    int maxIdArrLen = 1024;
    auto result = model.tokenize("a photo of a really, functistaner big cat.", maxIdArrLen);
    std::vector<int64_t> expected = {49406, 320,    1125,   539,   320, 1414,   267, 8679,  555,    2203,   528,    1205,   2368,   269,        49407};
    ASSERT_EQ(result.size(), expected.size());
    for (int i = 0; i < result.size(); i++) {
        EXPECT_EQ(result[i], expected[i]) << "expected: " << expected[i] << "; actual: " << result[i];
    }
}

// TODO: execute tests with context_length alignment
