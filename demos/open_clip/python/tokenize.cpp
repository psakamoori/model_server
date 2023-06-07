#include <iostream>
#include <vector>
#include <string>
#include <cinttypes>
#include <unordered_map>
#include <map>
#include <regex>
#include <exception>
#include <set>
#include <sstream>
#include <fstream>
#include <map>
#include <algorithm>
#include <chrono>

#include <pcre.h>
#include "json.hpp"

using json = nlohmann::json;

std::vector<std::string> split_by_regex(const std::string& sentence) {
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

// a photo of a really, functistaner big cat.
/*
a
photo
of
a
really
,
functistaner
big
cat
.
*/

size_t split(const std::string &txt, std::vector<std::string> &strs, char ch)
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

class SimpleTokenizer {
    json vocab;
    std::map<std::pair<std::string, std::string>, int64_t> bpe_ranks;
    unsigned char dd[3] = {0xc2,0xac,0x00};
public:
    SimpleTokenizer(const std::string bpe_path = "bpe_simple_vocab_16e6.txt"/* TODO: special tokens? */) {

        // for (auto [k, v] : bytes_to_unicode2()) {
        //     std::cout << k << ";";
        //     for (int i = 0; i < v.size(); i++) {
        //         std::cout << (unsigned int)(unsigned char)v[i] << " ";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << (char*)dd << std::endl;
        //std::vector<std::string> merges;
        std::ifstream f("vocab.json");
        vocab = json::parse(f);

        std::ifstream infile(bpe_path);
        std::string line;
        int64_t line_no = -1;
        int64_t rank = 0;
        while (std::getline(infile, line))
        {
            line_no++;
            if (line_no == 0)
                continue;
            
            if (line_no == 49152-256-2+1) {
                break;
            }

            //merges.push_back(line);

            std::vector<std::string> tpl;
            split(line, tpl, ' ');
            if (tpl.size() != 2)
                throw std::logic_error("tpl size is not 2");
            
            this->bpe_ranks.emplace(std::make_pair(std::make_pair(tpl[0],tpl[1]), rank++));
        }

        // int i = 20;
        // for (const auto& [k, v] : this->bpe_ranks) {
        //     if (i-- == 0)
        //         break;
        //     std::cout << k.first << "---------" << k.second << "-------------" << v << std::endl;
        // }
    }
    ~SimpleTokenizer() {}

    std::set<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& word) {
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

    std::pair<std::string, std::string> get_pair_with_lowest_rank(const std::set<std::pair<std::string, std::string>>& pairs) {
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
    std::vector<std::string> bpe(const std::string& token) {
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

    std::vector<std::int64_t> encode(const std::string& sentence) {
        std::vector<std::int64_t> ids;

        auto tokens = split_by_regex(sentence);
        for (const auto& token : tokens) {
            //std::cout << "Processing: [" << token << "]...\n";
            auto bpe_v = bpe(token);
            //std::cout << "BPE: [";
            // func ti stan er</w>
            for (auto& b : bpe_v) {
                //std::cout << b << ";";
                ids.push_back(vocab.at(b));
            }
            //std::cout << "]" << std::endl;

            // TODO: UTF8 thing

        }
        //std::cout << std::endl;

        return ids;
    }
};


#define OVECCOUNT 30    /* Maximum number of capturing groups */

void tokenize(const std::vector<std::string>& sentences, std::vector<std::vector<std::int64_t>>& tokens, int context_length = 77) {
    SimpleTokenizer tok;

    for (int i = 0; i < 10000; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        tok.encode(sentences[0]);
        //for (const auto& sentence : sentences)
        //    tokens.push_back(tok.encode(sentence));
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
        elapsed).count();
        std::cout << "Elapsed microseconds: " << microseconds << std::endl;
    }
}

int main() {
    std::vector<std::string> sentences = {
        //"a photo of a really, functistaner big cat."
        "a photo of a really, big cat."
    };

    std::vector<std::vector<std::int64_t>> tokens;
    tokenize(sentences, tokens);

    for (size_t i = 0; i < sentences.size(); i++) {
        std::cout << "Sentence: \"" << sentences[0] << "\" => [";
        for (size_t j = 0; j < tokens[i].size(); j++) {
            std::cout << tokens[i][j] << ", ";
        }
        std::cout << "]\n";
    }

    return 0;
}
