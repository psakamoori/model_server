#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "detokenizer.hpp"

using namespace custom_nodes::detokenizer;

TEST(DetokenizerTest, Run) {
    Model model("../gpt2.i2w");
    auto result = model.detokenizeEx({23294, 241, 22174, 28618, 2515, 94, 31676}, 1024);
    ASSERT_EQ(result, "こんにちは");
}

void prepare(std::vector<float> data, std::vector<size_t> shape, std::vector<std::vector<int64_t>> previousTokens, struct CustomNodeTensor* tensors) {
    // logits
    struct CustomNodeTensor* tensor = &tensors[0];
    tensor->dataBytes = data.size() * sizeof(float);
    tensor->data = (uint8_t*)malloc(tensor->dataBytes);
    std::memcpy(tensor->data, reinterpret_cast<uint8_t*>(data.data()), tensor->dataBytes);

    tensor->dimsCount = shape.size();
    tensor->dims = (uint64_t*)malloc(tensor->dimsCount * sizeof(uint64_t));
    int i = 0;
    for (size_t dim : shape) {
        tensor->dims[i] = dim;
        i++;
    }

    tensor->precision = FP32;
    tensor->name = "logits";
    // [2, 8, 50400]

    // input_ids
    tensor = &tensors[1];
    tensor->dataBytes = shape[0] * shape[1] * sizeof(int64_t);
    tensor->data = (uint8_t*)malloc(tensor->dataBytes);
    for (int i = 0; i < shape[0]; i++) {
        std::memcpy(
            tensor->data + i * shape[1] * sizeof(int64_t),
            reinterpret_cast<uint8_t*>(previousTokens[i].data()), previousTokens[i].size() * sizeof(int64_t));
    }

    tensor->dimsCount = 2;
    tensor->dims = (uint64_t*)malloc(tensor->dimsCount * sizeof(uint64_t));
    tensor->dims[0] = shape[0];
    tensor->dims[1] = shape[1];

    tensor->precision = I64;
    tensor->name = "input_ids";
    // [2, 8]

    // attention_mask
    tensor = &tensors[2];
    tensor->dataBytes = shape[0] * shape[1] * sizeof(int64_t);
    tensor->data = (uint8_t*)malloc(tensor->dataBytes);
    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            if (j < previousTokens[i].size()) {
                reinterpret_cast<int64_t*>(tensor->data)[i * shape[1] + j] = 1;
            } else {
                reinterpret_cast<int64_t*>(tensor->data)[i * shape[1] + j] = 0;
            }
        }
    }

    tensor->dimsCount = 2;
    tensor->dims = (uint64_t*)malloc(tensor->dimsCount * sizeof(uint64_t));
    tensor->dims[0] = shape[0];
    tensor->dims[1] = shape[1];

    tensor->precision = I64;
    tensor->name = "attention_mask";
    // [2, 8]
}

class DetokenizerFixtureTest : public ::testing::Test {
protected:
    void run(std::vector<float> data, std::vector<size_t> shape, std::vector<std::vector<int64_t>> previousTokens, std::vector<std::string>& out) {
        ASSERT_EQ(shape.size(), 3);
        struct CustomNodeTensor inputs[3];
        struct CustomNodeTensor* outputs = nullptr;
        int outputsCount = 0;
        prepare(data, shape, previousTokens, inputs);
        int ret = execute(inputs, 3, &outputs, &outputsCount, params, 2, model);
        for (int i = 0; i < 3; i++) {
            free(inputs[i].data);
            free(inputs[i].dims);
        }
        ASSERT_EQ(ret, 0);
        ASSERT_EQ(outputsCount, 1);
        std::vector<std::string> results;
        results.resize(outputs->dims[0]);
        for (int i = 0; i < outputsCount; i++) {
            if (std::strcmp(outputs[i].name, "texts") == 0) {
                for (int j = 0; j < outputs[i].dims[0]; j++) {
                    char* str = (char*)outputs[i].data + j * outputs[i].dims[1];
                    results[j] = std::string(str);
                }
            } else {
                FAIL() << "Unknown output name: " << outputs[i].name;
            }
        }
        out = results;
        ASSERT_EQ(release(outputs, model), 0);
    }
    void SetUp() override {
        params[0].key = "model_path";
        params[0].value = "../gpt2.i2w";
        params[1].key = "max_buffer_length";
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

TEST_F(DetokenizerFixtureTest, execute) {
    std::vector<std::string> outputs;
    run({1.0, 2.0, 3.0, 1.5}, {1,1,4}, {{18435}}, outputs);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0], "Hello#");

    outputs.clear();
    run({9.4, 0.2, -0.82, -0.74, 4.2, 1.9, 0.2, 0.95, /**/1.0, 2.0, 3.0, 1.5/**/}, {1,3,4}, {{23294, 241, 22174}}, outputs);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0], "こん#");

    outputs.clear();
    run({9.4, 0.2, -0.82, -0.74, 4.2, 1.9, 12.2, 0.95, /**/0.46, 1.18, 1.16, 1.02/**/}, {1,3,4}, {{23294, 241, 22174}}, outputs);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0], "こん\"");

    outputs.clear();
    run({9.4, 0.2, -0.82, -0.74, /*start 0*/0.46, 1.18, 1.16, 1.02/*end 0*/, 4.2, 1.9, 0.2, 0.95, /*start 1*/1.0, 2.0, 3.0, 1.5/*end 1*/}, {2,2,4}, {{18435, 995}, {18435, 995}}, outputs);
    ASSERT_EQ(outputs.size(), 2);
    ASSERT_EQ(outputs[0], "Hello world\"");
    ASSERT_EQ(outputs[1], "Hello world#");

    outputs.clear();
    run({9.4, 0.2, -0.82, -0.74, /*start 0*/0.46, 1.18, 1.16, 1.02/*end 0*/, /*start 1*/4.2, 1.9, 0.2, 0.95,/*end 1*/ 1.0, 2.0, 3.0, 1.5}, {2,2,4}, {{18435, 995}, {18435}}, outputs);
    ASSERT_EQ(outputs.size(), 2);
    ASSERT_EQ(outputs[0], "Hello world\"");
    ASSERT_EQ(outputs[1], "Hello!");
}
