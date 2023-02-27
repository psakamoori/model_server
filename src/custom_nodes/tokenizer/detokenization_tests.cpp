#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "detokenizer.hpp"

using namespace custom_nodes::detokenizer;

TEST(TokenizerTest, Run) {
    Model model("../gpt2.i2w");
    auto result = model.detokenizeEx({23294, 241, 22174, 28618, 2515, 94, 31676}, 1024);
    ASSERT_EQ(result, "こんにちは");
}
