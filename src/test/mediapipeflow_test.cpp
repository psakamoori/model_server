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
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../config.hpp"
#include "../grpcservermodule.hpp"
#include "../http_rest_api_handler.hpp"
#include "../kfs_frontend/kfs_grpc_inference_service.hpp"
#include "../mediapipe_internal/mediapipegraphdefinition.hpp"
#include "../metric_config.hpp"
#include "../metric_module.hpp"
#include "../model_service.hpp"
#include "../precision.hpp"
#include "../servablemanagermodule.hpp"
#include "../server.hpp"
#include "../shape.hpp"
#include "test_utils.hpp"

using namespace ovms;

using testing::HasSubstr;
using testing::Not;

class MediapipeFlowTest : public ::testing::TestWithParam<std::string> {
protected:
    ovms::Server& server = ovms::Server::instance();

    const Precision precision = Precision::FP32;
    std::unique_ptr<std::thread> t;
    std::string port = "9178";
    void SetUpServer(const char* configPath) {
        server.setShutdownRequest(0);
        randomizePort(this->port);
        char* argv[] = {(char*)"ovms",
            (char*)"--config_path",
            (char*)configPath,
            (char*)"--port",
            (char*)port.c_str()};
        int argc = 5;
        t.reset(new std::thread([&argc, &argv, this]() {
            EXPECT_EQ(EXIT_SUCCESS, server.start(argc, argv));
        }));
        auto start = std::chrono::high_resolution_clock::now();
        while ((server.getModuleState(SERVABLE_MANAGER_MODULE_NAME) != ovms::ModuleState::INITIALIZED) &&
               (!server.isReady()) &&
               (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() < 5)) {
        }
    }

    void SetUp() override {
    }
    void TearDown() {
        server.setShutdownRequest(1);
        t->join();
        server.setShutdownRequest(0);
    }
};

class MediapipeFlowAddTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_add_adapter_full.json");
    }
};
class MediapipeFlowDummyTest : public MediapipeFlowTest {
public:
    void SetUp() {
        SetUpServer("/ovms/src/test/mediapipe/config_mediapipe_dummy_adapter_full.json");
    }
};

TEST_P(MediapipeFlowDummyTest, Infer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;

    const std::string modelName = GetParam();
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{{"in", {DUMMY_MODEL_SHAPE, precision}}};
    preparePredictRequest(request, inputsMeta);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    // TODO validate output
    auto outputs = response.outputs();
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].name(), "out");
    ASSERT_EQ(outputs[0].shape().size(), 2);
    ASSERT_EQ(outputs[0].shape()[0], 1);
    ASSERT_EQ(outputs[0].shape()[1], 10);
}
TEST_P(MediapipeFlowAddTest, Infer) {
    const ovms::Module* grpcModule = server.getModule(ovms::GRPC_SERVER_MODULE_NAME);
    KFSInferenceServiceImpl& impl = dynamic_cast<const ovms::GRPCServerModule*>(grpcModule)->getKFSGrpcImpl();
    ::KFSRequest request;
    ::KFSResponse response;
    const std::string modelName = GetParam();
    request.Clear();
    response.Clear();
    inputs_info_t inputsMeta{
        {"in1", {DUMMY_MODEL_SHAPE, precision}},
        {"in2", {DUMMY_MODEL_SHAPE, precision}}};
    preparePredictRequest(request, inputsMeta);
    request.mutable_model_name()->assign(modelName);
    ASSERT_EQ(impl.ModelInfer(nullptr, &request, &response).error_code(), grpc::StatusCode::OK);
    auto outputs = response.outputs();
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(outputs[0].name(), "out");
    ASSERT_EQ(outputs[0].shape().size(), 2);
    ASSERT_EQ(outputs[0].shape()[0], 1);
    ASSERT_EQ(outputs[0].shape()[1], 10);
}

TEST(Mediapipe, MetadataDummy) {
    ConstructorEnabledModelManager manager;
    ovms::MediapipeGraphConfig mgc{"/ovms/src/test/mediapipe/graphdummy.pbtxt"};
    ovms::MediapipeGraphDefinition mediapipeDummy("mediapipeDummy", mgc);
    ASSERT_EQ(mediapipeDummy.validate(manager), StatusCode::OK);
    tensor_map_t inputs = mediapipeDummy.getInputsInfo();
    tensor_map_t outputs = mediapipeDummy.getOutputsInfo();
    ASSERT_EQ(inputs.size(), 1);
    ASSERT_EQ(outputs.size(), 1);
    ASSERT_NE(inputs.find("in"), inputs.end());
    ASSERT_NE(outputs.find("out"), outputs.end());
    const auto& input = inputs.at("in");
    EXPECT_EQ(input->getShape(), Shape({}));
    EXPECT_EQ(input->getPrecision(), ovms::Precision::UNDEFINED);
    const auto& output = outputs.at("out");
    EXPECT_EQ(output->getShape(), Shape({}));
    EXPECT_EQ(output->getPrecision(), ovms::Precision::UNDEFINED);
}

const std::vector<std::string> mediaGraphsDummy{"mediapipeDummy",
    "mediapipeDummyADAPTFULL"};
const std::vector<std::string> mediaGraphsAdd{"mediapipeAdd",
    "mediapipeAddADAPTFULL"};

class MediapipeConfig : public MediapipeFlowTest {
public:
    void TearDown() override {}
};

const std::string NAME = "Name";
TEST_F(MediapipeConfig, MediapipeGraphDefinitionNonExistentFile) {
    ConstructorEnabledModelManager manager;
    MediapipeGraphConfig mgc{"/ovms/NONEXISTENT_FILE"};
    MediapipeGraphDefinition mgd(NAME, mgc);
    EXPECT_EQ(mgd.validate(manager), StatusCode::FILE_INVALID);
}

TEST_F(MediapipeConfig, MediapipeAdd) {
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile("/ovms/src/test/mediapipe/config_mediapipe_add_adapter_full.json");
    EXPECT_EQ(status, ovms::StatusCode::OK);

    for (auto& graphName : mediaGraphsAdd) {
        auto graphDefinition = manager.getMediapipeFactory().findDefinitionByName(graphName);
        EXPECT_NE(graphDefinition, nullptr);
        EXPECT_EQ(graphDefinition->getStatus().isAvailable(), true);
    }

    manager.join();
}

TEST_F(MediapipeConfig, MediapipeFullRelativePaths) {
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile("/ovms/src/test/mediapipe/relative_paths/config_relative_dummy.json");
    EXPECT_EQ(status, ovms::StatusCode::OK);

    auto definitionAdd = manager.getMediapipeFactory().findDefinitionByName("mediapipeAddADAPT");
    EXPECT_NE(definitionAdd, nullptr);
    EXPECT_EQ(definitionAdd->getStatus().isAvailable(), true);

    auto definitionFull = manager.getMediapipeFactory().findDefinitionByName("mediapipeAddADAPTFULL");
    EXPECT_NE(definitionFull, nullptr);
    EXPECT_EQ(definitionFull->getStatus().isAvailable(), true);

    manager.join();
}

TEST_F(MediapipeConfig, MediapipeFullRelativePathsNegative) {
    ConstructorEnabledModelManager manager;
    auto status = manager.startFromFile("/ovms/src/test/mediapipe/relative_paths/config_relative_dummy_negative.json");
    EXPECT_EQ(status, ovms::StatusCode::OK);

    auto definitionAdd = manager.getMediapipeFactory().findDefinitionByName("mediapipeAddADAPT");
    EXPECT_NE(definitionAdd, nullptr);
    EXPECT_EQ(definitionAdd->getStatus().isAvailable(), false);

    auto definitionFull = manager.getMediapipeFactory().findDefinitionByName("mediapipeAddADAPTFULL");
    EXPECT_NE(definitionFull, nullptr);
    EXPECT_EQ(definitionFull->getStatus().isAvailable(), false);

    manager.join();
}

INSTANTIATE_TEST_SUITE_P(
    Test,
    MediapipeFlowAddTest,
    ::testing::ValuesIn(mediaGraphsAdd),
    [](const ::testing::TestParamInfo<MediapipeFlowTest::ParamType>& info) {
        return info.param;
    });
INSTANTIATE_TEST_SUITE_P(
    Test,
    MediapipeFlowDummyTest,
    ::testing::ValuesIn(mediaGraphsDummy),
    [](const ::testing::TestParamInfo<MediapipeFlowTest::ParamType>& info) {
        return info.param;
    });
