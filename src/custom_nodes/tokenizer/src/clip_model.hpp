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
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <cinttypes>
#include <unordered_map>
#include <map>
#include <exception>
#include <set>
#include <sstream>
#include <fstream>
#include <chrono>

#include "model.hpp"
#include "json.hpp"

namespace custom_nodes {
namespace tokenizer {

class ClipModel  {
    int id;
    bool debug;

    nlohmann::json vocab;
    std::map<std::pair<std::string, std::string>, int64_t> bpe_ranks;

    std::set<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& word);
    std::pair<std::string, std::string> get_pair_with_lowest_rank(const std::set<std::pair<std::string, std::string>>& pairs);
    std::vector<std::string> bpe(const std::string& token);
    std::vector<std::int64_t> encode(const std::string& sentence);

public:
    ClipModel(const std::string& vocabPath, const std::string& mergesPath, bool debug = false);
    ~ClipModel();

    std::vector<int64_t> tokenize(const std::string& text, int maxIdsArrLength);
    std::string detokenize(const std::vector<int64_t>& tokens, int maxBufferLength, bool skipSpecialTokens = false);
};

}  // namespace tokenizer
}  // namespace custom_nodes
