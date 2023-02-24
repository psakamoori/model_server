#include <gtest/gtest.h>

#include <cstring>

#include "tokenizer.hpp"

using namespace custom_nodes::tokenizer;

TEST(TokenizerTest, init_deinit) {
    void* model = nullptr;
    struct CustomNodeParam params[1];
    params[0].key = "model_path";
    params[0].value = "../gpt2.bin";
    int ret = initialize(&model, params, 1);
    ASSERT_EQ(ret, 0);
    ASSERT_NE(model, nullptr);

    ret = deinitialize(model);
    ASSERT_EQ(ret, 0);

    model = nullptr;
    params[0].value = "../invalid.bin";
    ret = initialize(&model, params, 1);
    ASSERT_NE(ret, 0);
    ASSERT_EQ(model, nullptr);

    ret = deinitialize(model);
    ASSERT_EQ(ret, 0);
}

TEST(TokenizerTest, inputs_info) {
    struct CustomNodeTensorInfo* info = nullptr;
    int infoCount = 0;
    struct CustomNodeParam params[1];
    params[0].key = "model_path";
    params[0].value = "../gpt2.bin";

    Model model(params[0].value);

    int ret = getInputsInfo(&info, &infoCount, params, 1, (void*)&model);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(infoCount, 1);
    ASSERT_EQ(std::strcmp(info[0].name, "texts"), 0);
    ASSERT_EQ(info[0].dimsCount, 2);
    ASSERT_EQ(info[0].dims[0], -1);
    ASSERT_EQ(info[0].dims[1], -1);
    ASSERT_EQ(info[0].precision, U8);
    ret = release(info, (void*)&model);
    ASSERT_EQ(ret, 0);
}

TEST(TokenizerTest, outputs_info) {
    struct CustomNodeTensorInfo* info = nullptr;
    int infoCount = 0;
    struct CustomNodeParam params[1];
    params[0].key = "model_path";
    params[0].value = "../gpt2.bin";

    Model model(params[0].value);

    int ret = getOutputsInfo(&info, &infoCount, params, 1, (void*)&model);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(infoCount, 2);

    ASSERT_EQ(std::strcmp(info[0].name, "tokens"), 0);
    ASSERT_EQ(info[0].dimsCount, 2);
    ASSERT_EQ(info[0].dims[0], -1);
    ASSERT_EQ(info[0].dims[1], -1);
    ASSERT_EQ(info[0].precision, I64);

    ASSERT_EQ(std::strcmp(info[1].name, "attention"), 0);
    ASSERT_EQ(info[1].dimsCount, 2);
    ASSERT_EQ(info[1].dims[0], -1);
    ASSERT_EQ(info[1].dims[1], -1);
    ASSERT_EQ(info[1].precision, I64);

    ret = release(info, (void*)&model);
    ASSERT_EQ(ret, 0);
}

TEST(TokenizerTest, execute) {
    FAIL() << "not implemented";

    /*
        // Empty strings
        // Normal texts
        // Japanese
    */
}
