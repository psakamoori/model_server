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
#include "clip_model.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include <vector>
#include <string>
#include <cinttypes>
#include <unordered_map>
#include <map>
#include <exception>
#include <set>
#include <sstream>
#include <fstream>
#include <chrono>

#include <pcre.h>
#include "json.hpp"

using json = nlohmann::json;

namespace custom_nodes {
namespace tokenizer {

static std::atomic<int> maxId{0};

static size_t split(const std::string &txt, std::vector<std::string> &strs, char ch)
{
    size_t pos = txt.find( ch );
    size_t initialPos = 0;
    strs.clear();

    // Decompose statement
    while( pos != std::string::npos ) {
        strs.push_back( txt.substr( initialPos, pos - initialPos ) );
        initialPos = pos + 1;

        pos = txt.find( ch, initialPos );
    }

    // Add the last one
    strs.push_back( txt.substr( initialPos, std::min( pos, txt.size() ) - initialPos + 1 ) );

    return strs.size();
}

static std::vector<std::string> split_by_regex(const std::string& sentence) {
    const char *pattern = R"('s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+)";
    const char *string = sentence.c_str();

    pcre *regex;
    const char *error;
    int erroffset;

    regex = pcre_compile(pattern, 0, &error, &erroffset, NULL);
    if (regex == NULL) {
        printf("Error compiling regex: %s\n", error);
        return {};
    }

    int ovector[30]; // ?
    int rc;
    int start_offset = 0;

    std::vector<std::string> results;
    while ((rc = pcre_exec(regex, NULL, string, strlen(string), start_offset, 0, ovector, 30)) >= 0) {
        for (int i = 0; i < rc; i++) {
            int start = ovector[2 * i];
            int end = ovector[2 * i + 1];
            // printf("Match: %.*s\n", end - start, string + start);
            std::string res;
            res.assign(string, start, end - start);
            results.push_back(std::move(res));
        }
        // printf("_______________________________\n");
        start_offset = ovector[1]; // Set the next search starting position
    }
    return results;
}

ClipModel::ClipModel(const std::string& vocabPath, const std::string& mergesPath, bool debug) :
    id(maxId++),
    debug(debug) {

    std::ifstream f(vocabPath);
    this->vocab = json::parse(f);  // TODO: Missing file handling


    std::ifstream infile(mergesPath); // TODO: Missing file handling
    std::string line;
    int64_t line_no = -1;
    int64_t rank = 0;
    while (std::getline(infile, line))
    {
        line_no++;
        if (line_no == 0)
            continue;
        
        if (line_no == 49152-256-2+1) {  // TODO: ?
            break;
        }

        std::vector<std::string> tpl;
        split(line, tpl, ' ');
        if (tpl.size() != 2)
            throw std::logic_error("tpl size is not 2");
        
        this->bpe_ranks.emplace(std::make_pair(std::make_pair(tpl[0],tpl[1]), rank++));
    }
}

ClipModel::~ClipModel() {
}

std::set<std::pair<std::string, std::string>> ClipModel::get_pairs(const std::vector<std::string>& word) {
    std::set<std::pair<std::string, std::string>> ret{};
    if (word.size() == 0)
        throw std::logic_error("word size 0");
    if (word.size() == 1)
        return ret;
    for (size_t i = 1; i < word.size(); i++) {
        ret.emplace(std::make_pair(word[i-1], word[i]));
    }
    return ret;
}

std::pair<std::string, std::string> ClipModel::get_pair_with_lowest_rank(const std::set<std::pair<std::string, std::string>>& pairs) {
    std::pair<std::string, std::string> p;
    int64_t minValue = std::numeric_limits<int64_t>::max();
    for (auto& pair : pairs) {
        try {
            int64_t v = this->bpe_ranks.at(pair);  // assuming it can be found TODO
            if (v < minValue) {
                minValue = v;
                p = pair;
            }
        } catch (...) {
            continue;
        }
    }
    if (minValue == std::numeric_limits<int64_t>::max()) {
        throw std::out_of_range("at");
    }
    return p;
}
// functistaner => func ti stan er</w>
std::vector<std::string> ClipModel::bpe(const std::string& token) {
    // TODO: Cache?
    if (token.size() == 0)
        throw std::logic_error("token size 0");

    // f u n c t i s t a n e r</w>
    std::vector<std::string> word;
    for (size_t i = 0; i < token.size(); i++) {
        if (i == token.size() - 1) {
            word.push_back(std::string(1, token[i]) + std::string("</w>"));
        } else {
            word.push_back(std::string(1, token[i]));
        }
    }
    // for (auto& s : word) {
    //     std::cout << s << "|";
    // }
    // std::cout << std::endl;
    auto pairs = get_pairs(word);
    // for (const auto& [k, v] : pairs) {
    //     std::cout << k << "=>" << v << std::endl;
    // }
    // std::cout << std::endl;
    if (pairs.size() == 0) {
        return std::vector{std::string(token)+std::string("</w>")};
    }
    while (true) {
        // get pair with lowest rank
        if (pairs.size() == 0)
            break;
        
        std::pair<std::string, std::string> pair_with_lowest_rank;
        try {
            pair_with_lowest_rank = get_pair_with_lowest_rank(pairs);
        } catch (std::out_of_range& e) {
            // std::cout << e.what() << std::endl;
            break;
        }
        // for (auto& g : word)
        //     std::cout << g << ",";
        // std::cout << std::endl;
        auto first = pair_with_lowest_rank.first;
        auto second = pair_with_lowest_rank.second;
        // a n
        // f u n c t i s t a n e r</w>
        // f u n c t i s t an e r</w>
        // f u n c t i s t an er</w>
        // f un c t i s t an er</w>
        // f unc t i s t an er</w>
        // func t i s t an er</w>
        // func ti s t an er</w>
        // func ti st an er</w>
        // func ti stan er</w>
        
        //std::cout << "Pair with lowest rank: (" << first << "," << second << ")" << std::endl;
        std::vector<std::string> new_word{};
        int64_t i = 0;
        while (i < word.size()) {
            //std::cout << "i: " << i << std::endl;
            ptrdiff_t pos = std::distance(word.begin(), std::find(word.begin() + i, word.end(), first));
            //std::cout << "i: " << i <<"; pos: " << pos << std::endl;
            // auto it = std::find(word.begin() + i, word.end(), first);
            if (pos >= word.size()) {
                for (int j = i; j < word.size(); j++)
                    new_word.push_back(word[j]);
                break;
            } else {
                for (int j = i; j < pos; j++)
                    new_word.push_back(word[j]);
                i = pos;
            }
            if (word[i] == first && i < word.size() - 1 && word[i+1] == second) {
                new_word.push_back(first + second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }
        word = new_word;
        if (word.size() == 1)
            break;
        else
            pairs = get_pairs(word);
    }
    return word;
}
std::vector<std::int64_t> ClipModel::encode(const std::string& sentence) {
    std::cout << "Tokenizing: [" << sentence << "]" << std::endl;
    std::vector<std::int64_t> ids;
    ids.push_back(vocab.at("<|startoftext|>"));
    auto tokens = split_by_regex(sentence);
    for (const auto& token : tokens) {
        std::cout << "Processing: [" << token << "]...\n";
        auto bpe_v = bpe(token);
        std::cout << "BPE: [";
        // func ti stan er</w>
        for (auto& b : bpe_v) {
            std::cout << b << ";";
            ids.push_back(vocab.at(b));
        }
        std::cout << "]" << std::endl;
        // TODO: UTF8 thing
    }
    std::cout << std::endl;
    ids.push_back(vocab.at("<|endoftext|>"));
    return ids;
}

std::vector<int64_t> ClipModel::tokenize(const std::string& text, int maxIdsArrLength/*unused*/) {
    // maxIdsArrLength TODO?
    return encode(text);
}

std::string ClipModel::detokenize(const std::vector<int64_t>& tokens, int maxBufferLength, bool skipSpecialTokens) {
    return std::string{};
}

}  // namespace tokenizer
}  // namespace custom_nodes
