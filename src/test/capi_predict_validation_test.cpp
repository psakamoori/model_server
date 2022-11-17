//*****************************************************************************
// Copyright 2022 Intel Corporation
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

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../inferencerequest.hpp"
#include "../modelconfig.hpp"
#include "../modelinstance.hpp"
#include "../predict_request_validation_utils.hpp"
#include "test_utils.hpp"

using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"

class MockModelInstance : public ovms::ModelInstance {
public:
    MockModelInstance(ov::Core& ieCore) :
        ModelInstance("UNUSED_NAME", 42, ieCore) {}
    MOCK_METHOD(const ovms::tensor_map_t&, getInputsInfo, (), (const, override));
    MOCK_METHOD(ovms::Dimension, getBatchSize, (), (const, override));
    MOCK_METHOD(const ovms::ModelConfig&, getModelConfig, (), (const, override));

    const ovms::Status mockValidate(const ovms::InferenceRequest* request) {
        return validate(request);
    }
};

class CAPIPredictValidation : public ::testing::Test {
protected:
    std::unique_ptr<ov::Core> ieCore;
    std::unique_ptr<NiceMock<MockModelInstance>> instance;
    ovms::InferenceRequest request{"model_name", 1};
    ovms::ModelConfig modelConfig{"model_name", "model_path"};
    ovms::tensor_map_t servableInputs;

    void SetUp() override {
        ieCore = std::make_unique<ov::Core>();
        instance = std::make_unique<NiceMock<MockModelInstance>>(*ieCore);

        servableInputs = ovms::tensor_map_t({
            {"Input_FP32_1_224_224_3_NHWC",
                std::make_shared<ovms::TensorInfo>("Input_FP32_1_3_224_224_NHWC", ovms::Precision::FP32, ovms::shape_t{1, 224, 224, 3}, ovms::Layout{"NHWC"})},
            {"Input_U8_1_3_62_62_NCHW",
                std::make_shared<ovms::TensorInfo>("Input_U8_1_3_62_62_NCHW", ovms::Precision::U8, ovms::shape_t{1, 3, 62, 62}, ovms::Layout{"NCHW"})},
            {"Input_I64_1_6_128_128_16_NCDHW",
                std::make_shared<ovms::TensorInfo>("Input_I64_1_6_128_128_16_NCDHW", ovms::Precision::I64, ovms::shape_t{1, 6, 128, 128, 16}, ovms::Layout{"NCDHW"})},
            {"Input_U16_1_2_8_4_NCHW",
                std::make_shared<ovms::TensorInfo>("Input_U16_1_2_8_4_NCHW", ovms::Precision::U16, ovms::shape_t{1, 2, 8, 4}, ovms::Layout{"NCHW"})},
        });

        ON_CALL(*instance, getInputsInfo()).WillByDefault(ReturnRef(servableInputs));
        ON_CALL(*instance, getBatchSize()).WillByDefault(Return(1));
        ON_CALL(*instance, getModelConfig()).WillByDefault(ReturnRef(modelConfig));

        preparePredictRequest(request,
            {{"Input_FP32_1_224_224_3_NHWC",
                 std::tuple<ovms::shape_t, ovms::Precision>{{1, 224, 224, 3}, ovms::Precision::FP32}},
                {"Input_U8_1_3_62_62_NCHW",
                    std::tuple<ovms::shape_t, ovms::Precision>{{1, 3, 62, 62}, ovms::Precision::U8}},
                {"Input_I64_1_6_128_128_16_NCDHW",
                    std::tuple<ovms::shape_t, ovms::Precision>{{1, 6, 128, 128, 16}, ovms::Precision::I64}},
                {"Input_U16_1_2_8_4_NCHW",
                    std::tuple<ovms::shape_t, ovms::Precision>{{1, 2, 8, 4}, ovms::Precision::U16}}});
    }
};

TEST_F(CAPIPredictValidation, ValidRequest) {
    auto status = instance->mockValidate(&request);
    EXPECT_TRUE(status.ok()) << status.string();
}

TEST_F(CAPIPredictValidation, RequestNotEnoughInputs) {
    //request.mutable_inputs()->RemoveLast();
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_INPUTS) << status.string();
}

TEST_F(CAPIPredictValidation, RequestTooManyInputs) {
    //auto inputWrongName = request.add_inputs();
    //inputWrongName->set_name("Some_Input");
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_INPUTS) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongInputName) {
    //request.mutable_inputs()->RemoveLast();  // remove redundant input
    //auto inputWrongName = request.add_inputs();
    //inputWrongName->set_name("Some_Input");
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_MISSING_INPUT) << status.string();
}

TEST_F(CAPIPredictValidation, RequestTooManyShapeDimensions) {
    //auto someInput = request.mutable_inputs()->Mutable(request.mutable_inputs()->size() - 1);  // modify last
    //someInput->mutable_shape()->Add(16);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS) << status.string();
}

TEST_F(CAPIPredictValidation, RequestNotEnoughShapeDimensions) {
    //auto someInput = request.mutable_inputs()->Mutable(request.mutable_inputs()->size() - 1);  // modify last
    //someInput->mutable_shape()->Clear();
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongBatchSize) {
    //auto someInput = request.mutable_inputs()->Mutable(request.mutable_inputs()->size() - 1);  // modify last
    //someInput->mutable_shape()->Set(0, 10);                                                    // dim(0) is batch size

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");
    //auto someInput = request.mutable_inputs()->Mutable(request.mutable_inputs()->size() - 1);  // modify last
    //someInput->mutable_shape()->Set(0, 10);                                                    // dim(0) is batch size. Change from 1
    //auto bufferId = request.mutable_inputs()->size() - 1;
    //auto previousSize = request.raw_input_contents()[bufferId].size();
    //request.mutable_raw_input_contents(bufferId)->assign(size_t(previousSize * 10), '1');
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED) << status.string();
}
/*
TEST_F(CAPIPredictValidation, ValidRequestBinaryInputs) {
    modelConfig.setBatchingParams("auto");
    std::string inputName = "Binary_Input";
    ovms::InferenceRequest binaryInputRequest;

    auto input = binaryInputRequest.add_inputs();
    input->set_name(inputName);
    input->set_datatype("BYTES");
    const int requestBatchSize = 1;
    std::string bytes_contents = "BYTES_CONTENTS";
    for (int i = 0; i < requestBatchSize; i++) {
        input->mutable_contents()->add_bytes_contents(bytes_contents.c_str(), bytes_contents.size());
    }
    input->mutable_shape()->Add(requestBatchSize);

    servableInputs.clear();
    ovms::shape_t shape = {1, 3, 224, 224};
    servableInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::FP32,
        shape,
        ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_TRUE(status.ok());
}

TEST_F(CAPIPredictValidation, RequestWrongBatchSizeBinaryInputs) {
    std::string inputName = "Binary_Input";
    ovms::InferenceRequest binaryInputRequest;

    auto input = binaryInputRequest.add_inputs();
    input->set_name(inputName);
    input->set_datatype("BYTES");
    const int requestBatchSize = 2;
    std::string bytes_contents = "BYTES_CONTENTS";
    for (int i = 0; i < requestBatchSize; i++) {
        input->mutable_contents()->add_bytes_contents(bytes_contents.c_str(), bytes_contents.size());
    }
    input->mutable_shape()->Add(requestBatchSize);

    servableInputs.clear();
    ovms::shape_t shape = {1, 3, 224, 224};
    servableInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::FP32,
        shape,
        ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE);
}

TEST_F(CAPIPredictValidation, RequestWrongBatchSizeAutoBinaryInputs) {
    modelConfig.setBatchingParams("auto");
    std::string inputName = "Binary_Input";
    ovms::InferenceRequest binaryInputRequest;

    auto input = binaryInputRequest.add_inputs();
    input->set_name(inputName);
    input->set_datatype("BYTES");
    const int requestBatchSize = 2;
    std::string bytes_contents = "BYTES_CONTENTS";
    for (int i = 0; i < requestBatchSize; i++) {
        input->mutable_contents()->add_bytes_contents(bytes_contents.c_str(), bytes_contents.size());
    }
    input->mutable_shape()->Add(requestBatchSize);

    servableInputs.clear();
    ovms::shape_t shape = {1, 3, 224, 224};
    servableInputs[inputName] = std::make_shared<ovms::TensorInfo>(
        inputName,
        ovms::Precision::FP32,
        shape,
        ovms::Layout{"NHWC"});

    auto status = instance->mockValidate(&binaryInputRequest);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);
}

TEST_F(CAPIPredictValidation, RequestWrongAndCorrectBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");

    // First is incorrect, second is correct
    preparePredictRequest(request, {{"im_data", {{3, 3, 800, 1344}, ovms::Precision::FP32}},
                                       {"im_info", {{1, 3}, ovms::Precision::FP32}}});

    servableInputs.clear();
    servableInputs = ovms::tensor_map_t{
        {"im_data", std::make_shared<ovms::TensorInfo>("im_data", ovms::Precision::FP32, ovms::shape_t{1, 3, 800, 1344}, ovms::Layout{"NCHW"})},
        {"im_info", std::make_shared<ovms::TensorInfo>("im_info", ovms::Precision::FP32, ovms::shape_t{1, 3}, ovms::Layout{"NC"})},
    };

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED);

    preparePredictRequest(request, {{"im_data", {{1, 3, 800, 1344}, ovms::Precision::FP32}},
                                       {"im_info", {{3, 3}, ovms::Precision::FP32}}});

    status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::BATCHSIZE_CHANGE_REQUIRED) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongAndCorrectShapeAuto) {
    modelConfig.parseShapeParameter("auto");
    preparePredictRequest(request, {{"im_data", {{1, 3, 900, 1344}, ovms::Precision::FP32}},
                                       {"im_info", {{1, 3}, ovms::Precision::FP32}}});

    // First is incorrect, second is correct
    servableInputs.clear();
    servableInputs = ovms::tensor_map_t{
        {"im_data", std::make_shared<ovms::TensorInfo>("im_data", ovms::Precision::FP32, ovms::shape_t{1, 3, 800, 1344}, ovms::Layout{"NCHW"})},
        {"im_info", std::make_shared<ovms::TensorInfo>("im_info", ovms::Precision::FP32, ovms::shape_t{1, 3}, ovms::Layout{"NC"})},
    };

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED) << status.string();

    // First is correct, second is incorrect
    preparePredictRequest(request, {{"im_data", {{1, 3, 800, 1344}, ovms::Precision::FP32}},
                                       {"im_info", {{1, 6}, ovms::Precision::FP32}}});

    status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED) << status.string();
}
TEST_F(CAPIPredictValidation, RequestValidBatchSizeAuto) {
    modelConfig.setBatchingParams("auto");
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::OK) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongShapeValues) {
    auto& input = (*request.mutable_inputs(request.inputs().size() - 1));
    input.mutable_shape()->RemoveLast();
    input.mutable_shape()->Add(12345);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongShapeValuesTwoInputsOneWrong) {  // one input fails validation, request denied
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"auto\"}");
    auto& input = (*request.mutable_inputs(request.inputs().size() - 1));
    input.mutable_shape()->RemoveLast();
    input.mutable_shape()->Add(123);
    auto& input2 = (*request.mutable_inputs(request.inputs().size() - 2));
    input2.mutable_shape()->RemoveLast();
    input2.mutable_shape()->Add(123);

    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE) << status.string();
}*/
/*
TEST_F(CAPIPredictValidation, RequestWrongShapeValuesAuto) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"auto\"}");
    prepareKFSInferInputTensor(request, "Input_U8_1_3_62_62_NCHW", {{1, 4, 64, 64}, "UINT8"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongShapeValuesAutoTwoInputs) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"auto\", \"Input_U16_1_2_8_4_NCHW\": \"auto\"}");
    prepareKFSInferInputTensor(request, "Input_U8_1_3_62_62_NCHW", {{1, 4, 63, 63}, "UINT8"});
    prepareKFSInferInputTensor(request, "Input_U16_1_2_8_4_NCHW", {{1, 2, 16, 8}, "UINT16"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);
}

TEST_F(CAPIPredictValidation, RequestWrongShapeValuesAutoNoNamedInput) {
    modelConfig.parseShapeParameter("auto");
    prepareKFSInferInputTensor(request, "Input_U8_1_3_62_62_NCHW", {{1, 4, 63, 63}, "UINT8"});
    prepareKFSInferInputTensor(request, "Input_U16_1_2_8_4_NCHW", {{1, 2, 16, 8}, "UINT16"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED);
}

TEST_F(CAPIPredictValidation, RequestWrongShapeValuesAutoFirstDim) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"auto\"}");
    prepareKFSInferInputTensor(request, "Input_U8_1_3_62_62_NCHW", {{2, 3, 62, 62}, "UINT8"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::RESHAPE_REQUIRED) << status.string();
}

TEST_F(CAPIPredictValidation, RequestValidShapeValuesTwoInputsFixed) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"(1,3,62,62)\", \"Input_U16_1_2_8_4_NCHW\": \"(1,2,8,4)\"}");
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::OK) << status.string();
}

TEST_F(CAPIPredictValidation, RequestWrongShapeValuesFixed) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"(1,3,62,62)\"}");
    prepareKFSInferInputTensor(request, "Input_U8_1_3_62_62_NCHW", {{1, 4, 63, 63}, "UINT8"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_SHAPE) << status.string();
}
TEST_F(CAPIPredictValidation, RequestWrongShapeValuesFixedFirstDim) {
    modelConfig.parseShapeParameter("{\"Input_U8_1_3_62_62_NCHW\": \"(1,3,62,62)\"}");
    prepareKFSInferInputTensor(request, "Input_U8_1_3_62_62_NCHW", {{2, 3, 62, 62}, "UINT8"});
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_BATCH_SIZE) << status.string();
}

TEST_F(CAPIPredictValidation, RequestIncorrectContentSize) {
    findKFSInferInputTensorContentInRawInputs(request, "Input_I64_1_6_128_128_16_NCDHW")->assign('c', 2);
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE) << status.string();
}

TEST_F(CAPIPredictValidation, RequestIncorrectContentSizeBatchAuto) {
    modelConfig.setBatchingParams("auto");
    prepareKFSInferInputTensor(request, "Input_I64_1_6_128_128_16_NCDHW", {{1, 6, 128, 128, 16}, "INT64"});
    auto input = findKFSInferInputTensor(request, "Input_I64_1_6_128_128_16_NCDHW");
    (*input->mutable_shape()->Mutable(0)) = 2;
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE) << status.string();
}

TEST_F(CAPIPredictValidation, RequestIncorrectContentSizeShapeAuto) {
    modelConfig.parseShapeParameter("auto");
    prepareKFSInferInputTensor(request, "Input_I64_1_6_128_128_16_NCDHW", {{1, 6, 128, 128, 16}, "INT64"});
    auto input = findKFSInferInputTensor(request, "Input_I64_1_6_128_128_16_NCDHW");
    (*input->mutable_shape()->Mutable(1)) = 2;
    auto status = instance->mockValidate(&request);
    EXPECT_EQ(status, ovms::StatusCode::INVALID_CONTENT_SIZE) << status.string();
}
*/
#pragma GCC diagnostic pop