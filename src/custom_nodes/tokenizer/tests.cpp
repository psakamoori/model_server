#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

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

void putStringsToTensor(std::vector<std::string> strings, struct CustomNodeTensor& tensor) {
    size_t maxStringLength = 0;
    for (auto& str : strings) {
        maxStringLength = std::max(str.size(), maxStringLength);
    }
    size_t width = maxStringLength + 1;

    tensor.dataBytes = strings.size() * width * sizeof(uint8_t);
    tensor.data = (uint8_t*)malloc(tensor.dataBytes);
    
    int i = 0;
    for (auto& str : strings) {
        std::memcpy(tensor.data + i * width, str.c_str(), str.size());
        tensor.data[i * width + str.size()] = 0;
        i++; 
    }

    tensor.dimsCount = 2;
    tensor.dims = (uint64_t*)malloc(2 * sizeof(uint64_t));
    tensor.dims[0] = strings.size();
    tensor.dims[1] = width;

    tensor.precision = U8;
    tensor.name = "texts";
}

class TokenizerFixtureTest : public ::testing::Test {
protected:
    struct output {
        std::vector<int64_t> tokens;
        std::vector<int64_t> attention;
    };
    void run(std::vector<std::string> in, std::vector<output>& out) {
        struct CustomNodeTensor inputs[1];
        struct CustomNodeTensor* outputs = nullptr;
        int outputsCount = 0;
        putStringsToTensor(in, inputs[0]);
        int ret = execute(inputs, 1, &outputs, &outputsCount, params, 2, model);
        free(inputs[0].data);
        free(inputs[0].dims);
        ASSERT_EQ(ret, 0);
        ASSERT_EQ(outputsCount, 2);
        std::vector<output> result;
        result.resize(outputs->dims[0]);
        for (int i = 0; i < outputsCount; i++) {
            if (std::strcmp(outputs[i].name, "attention") == 0) {
                // in: [2, 80]; out: [2, 144]
                for (int i = 0; i < outputs[i].dims[0]; i++) {
                    result[i].attention = std::vector<int64_t>(
                        (int64_t*)outputs[i].data,
                        (int64_t*)outputs[i].data + (outputs[i].dataBytes / sizeof(int64_t)));
                }
            } else if (std::strcmp(outputs[i].name, "tokens") != 0) {
                for (int i = 0; i < outputs[i].dims[0]; i++) {
                    result[i].tokens = std::vector<int64_t>(
                        (int64_t*)outputs[i].data,
                        (int64_t*)outputs[i].data + (outputs[i].dataBytes / sizeof(int64_t)));
                }
            }
        }
        out = result;
        ASSERT_EQ(release(outputs, model), 0);
    }
    void SetUp() override {
        params[0].key = "model_path";
        params[0].value = "../gpt2.bin";
        params[1].key = "max_ids_arr_length";
        params[1].value = "1024";
        int ret = initialize(&model, params, 2);
        ASSERT_EQ(ret, 0);
        ASSERT_NE(model, nullptr);
    }
    void TearDown() override {
        int ret = deinitialize(model);
        ASSERT_EQ(ret, 0);
    }
    struct CustomNodeParam params[2];
    void* model = nullptr;
};

TEST_F(TokenizerFixtureTest, execute) {
    std::vector<output> outputs;
    run({"", "Hello world!", "こんにちは"}, outputs);
    ASSERT_EQ(outputs.size(), 3);

    /*
        // Empty strings
        // Normal texts
        // Japanese
    */

   /*
    // Negative params?
   */
}
