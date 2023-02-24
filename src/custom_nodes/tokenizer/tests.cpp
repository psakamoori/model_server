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

std::vector<int64_t> getTokensFromOutput(struct CustomNodeTensor* outputs, int outputsCount) {
    struct CustomNodeTensor* textTensor = nullptr;
    for (int i = 0; i < outputsCount; i++) {
        if (std::strcmp(outputs[i].name, "tokens") == 0) {
            textTensor = &(outputs[i]);
        }
    }
    if (textTensor == nullptr) {
        throw std::runtime_error("tokens tensor not found");
    }
    return std::vector<int64_t>((int64_t*)textTensor->data, (int64_t*)textTensor->data + (textTensor->dataBytes / sizeof(int64_t)));
}

TEST(TokenizerTest, execute) {

/*
int execute(
    const struct CustomNodeTensor* inputs,
    int inputsCount,
    struct CustomNodeTensor** outputs,
    int* outputsCount,
    const struct CustomNodeParam* params,
    int paramsCount,
    void* customNodeLibraryInternalManager);
*/
    struct CustomNodeTensor inputs[1];
    struct CustomNodeTensor* outputs = nullptr;
    int outputsCount = 0;
    struct CustomNodeParam params[2];
    params[0].key = "model_path";
    params[0].value = "../gpt2.bin";
    params[1].key = "max_ids_arr_length";
    params[1].value = "1024";
    Model model(params[0].value);

    std::vector<std::string> texts = {
        "Эpple pie. How do I renew my virtual smart card?: /Microsoft IT/ 'virtual' smart card certificates for DirectAccess are valid for one year. In order to get to microsoft.com we need to type pi@1.2.1.2.",
        // "This is a test with a longer text"
    };

    std::vector<std::vector<int64_t>> expectedTokens = {
        {1208,  9397,  2571, 11345,  1012,  2129,  2079, 1045, 20687, 2026,  7484,  6047,
            4003,  1029,  1024,  1013,  7513,  2009,  1013,  1005,  7484,  1005,  6047,  4003,
            17987,  2005,  3622,  6305,  9623,  2015,  2024,  9398,  2005,  2028,  2095,  1012,
            1999,  2344,  2000,  2131,  2000,  7513,  1012,  4012,  2057,  2342,  2000,  2828,
            14255,  1030,  1015,  1012,  1016,  1012,  1015,  1012,  1016,  1012},
        // {}
    };

    putStringsToTensor(texts, inputs[0]);
    inputs[0].name = "texts";
    int ret = execute(inputs, 1, &outputs, &outputsCount, params, 2, (void*)&model);
    ASSERT_EQ(ret, 0);
    // ASSERT_EQ(release(inputs, (void*)&model), 0);
    auto tokens = getTokensFromOutput(outputs, outputsCount);
    ASSERT_EQ(tokens, expectedTokens[0]);
    /*
        // Empty strings
        // Normal texts
        // Japanese
    */

   /*
    // Negative params?
   */
}
