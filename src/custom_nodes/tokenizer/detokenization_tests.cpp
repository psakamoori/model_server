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

void prepare(std::vector<float> data, std::vector<size_t> shape, struct CustomNodeTensor& tensor) {
    tensor.dataBytes = data.size() * sizeof(float);
    tensor.data = (uint8_t*)malloc(tensor.dataBytes);
    std::memcpy(tensor.data, reinterpret_cast<uint8_t*>(data.data()), tensor.dataBytes) ;

    tensor.dimsCount = shape.size();
    tensor.dims = (uint64_t*)malloc(tensor.dimsCount * sizeof(uint64_t));
    int i = 0;
    for (size_t dim : shape) {
        tensor.dims[i] = dim;
        i++;
    }

    tensor.precision = FP32;
    tensor.name = "logits";
}

class DetokenizerFixtureTest : public ::testing::Test {
protected:
    void run(std::vector<float> data, std::vector<size_t> shape, std::vector<std::string>& out) {
        struct CustomNodeTensor inputs[1];
        struct CustomNodeTensor* outputs = nullptr;
        int outputsCount = 0;
        prepare(data, shape, inputs[0]);
        int ret = execute(inputs, 1, &outputs, &outputsCount, params, 2, model);
        free(inputs[0].data);
        free(inputs[0].dims);
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
    run({1.0, 2.0, 3.0, 1.5}, {1,1,4}, outputs);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0], "#");

    outputs.clear();
    run({9.4, 0.2, -0.82, -0.74, 4.2, 1.9, 0.2, 0.95, /**/1.0, 2.0, 3.0, 1.5/**/}, {1,3,4}, outputs);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0], "#");

    outputs.clear();
    run({9.4, 0.2, -0.82, -0.74, 4.2, 1.9, 12.2, 0.95, /**/0.46, 1.18, 1.16, 1.02/**/}, {1,3,4}, outputs);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0], "\"");

    outputs.clear();
    run({9.4, 0.2, -0.82, -0.74, /*start 0*/0.46, 1.18, 1.16, 1.02/*end 0*/, 4.2, 1.9, 0.2, 0.95, /*start 1*/1.0, 2.0, 3.0, 1.5/*end 1*/}, {2,2,4}, outputs);
    ASSERT_EQ(outputs.size(), 2);
    ASSERT_EQ(outputs[0], "\"");
    ASSERT_EQ(outputs[1], "#");
}
