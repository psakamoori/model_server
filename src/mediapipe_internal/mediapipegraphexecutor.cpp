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
#include "mediapipegraphexecutor.hpp"

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../deserialization.hpp"
#include "../execution_context.hpp"
#include "../kfs_frontend/kfs_utils.hpp"
#include "../metric.hpp"
#include "../modelmanager.hpp"
#include "../serialization.hpp"
#include "../status.hpp"
#include "../stringutils.hpp"
#include "../tensorinfo.hpp"
#include "../tfs_frontend/tfs_utils.hpp"
#include "../timer.hpp"
#include "../version.hpp"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/image_opencv.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipegraphdefinition.hpp"  // for version in response
#include "opencv2/opencv.hpp"

namespace ovms {
static Status getRequestInput(google::protobuf::internal::RepeatedPtrIterator<const inference::ModelInferRequest_InferInputTensor>& itr, const std::string& requestedName, const KFSRequest& request) {
    auto requestInputItr = std::find_if(request.inputs().begin(), request.inputs().end(), [&requestedName](const ::KFSRequest::InferInputTensor& tensor) { return tensor.name() == requestedName; });
    if (requestInputItr == request.inputs().end()) {
        std::stringstream ss;
        ss << "Required input: " << requestedName;
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Missing input with specific name - {}", request.model_name(), request.model_version(), details);
        return Status(StatusCode::INVALID_MISSING_INPUT, details);
    }
    itr = requestInputItr;
    return StatusCode::OK;
}

static Status deserializeTensor(const std::string& requestedName, const KFSRequest& request, tensorflow::Tensor& outTensor) {
    using tensorflow::Tensor;
    using tensorflow::TensorShape;
    auto requestInputItr = request.inputs().begin();
    auto status = getRequestInput(requestInputItr, requestedName, request);
    if (!status.ok()) {
        return status;
    }
    // TODO there is no sense to check this for every input
    if (request.raw_input_contents().size() == 0 || request.raw_input_contents().size() != request.inputs().size()) {
        std::stringstream ss;
        ss << "Cannot find data in raw_input_content for input with name: " << requestedName;
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid message structure - {}", request.model_name(), request.model_version(), details);
        return Status(StatusCode::INVALID_MESSAGE_STRUCTURE, details);
    }
    auto inputIndex = requestInputItr - request.inputs().begin();
    auto& bufferLocation = request.raw_input_contents().at(inputIndex);
    try {
        auto datatype = getPrecisionAsDataType(KFSPrecisionToOvmsPrecision(requestInputItr->datatype()));
        TensorShape tensorShape;
        std::vector<int64_t> rawShape;
        for (int i = 0; i < requestInputItr->shape().size(); i++) {
            if (requestInputItr->shape()[i] <= 0) {
                std::stringstream ss;
                ss << "Negative or zero dimension size is not acceptable: " << tensorShapeToString(requestInputItr->shape()) << "; input name: " << requestedName;
                const std::string details = ss.str();
                SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", request.model_name(), request.model_version(), details);
                return Status(StatusCode::INVALID_SHAPE, details);
            }
            rawShape.emplace_back(requestInputItr->shape()[i]);
        }
        int64_t dimsCount = rawShape.size();
        tensorflow::TensorShapeUtils::MakeShape(rawShape.data(), dimsCount, &tensorShape);
        TensorShape::BuildTensorShapeBase(rawShape, static_cast<tensorflow::TensorShapeBase<TensorShape>*>(&tensorShape));
        // TODO here we allocate default TF CPU allocator
        tensorflow::Tensor localTensor(datatype, tensorShape);  // TODO error handling
        void* tftensordata = localTensor.data();
        std::memcpy(tftensordata, bufferLocation.data(), bufferLocation.size());
        outTensor = std::move(localTensor);
    } catch (const std::exception& e) {
        SPDLOG_DEBUG("Exception: {}; caught during Mediapipe TF tensor deserialization", e.what());
    } catch (...) {
        SPDLOG_ERROR("Unknown exception caught during Mediapipe TF tensor deserialization");
    }
    return StatusCode::OK;
}

static Status deserializeTensor(const std::string& requestedName, const KFSRequest& request, ov::Tensor& outTensor) {
    auto requestInputItr = request.inputs().begin();
    auto status = getRequestInput(requestInputItr, requestedName, request);
    if (!status.ok()) {
        return status;
    }
    // TODO there is no sense to check this for every input
    if (request.raw_input_contents().size() == 0 || request.raw_input_contents().size() != request.inputs().size()) {
        std::stringstream ss;
        ss << "Cannot find data in raw_input_content for input with name: " << requestedName;
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid message structure - {}", request.model_name(), request.model_version(), details);
        return Status(StatusCode::INVALID_MESSAGE_STRUCTURE, details);
    }
    auto inputIndex = requestInputItr - request.inputs().begin();
    auto& bufferLocation = request.raw_input_contents().at(inputIndex);
    try {
        ov::Shape shape;
        for (int i = 0; i < requestInputItr->shape().size(); i++) {
            if (requestInputItr->shape()[i] <= 0) {
                std::stringstream ss;
                ss << "Negative or zero dimension size is not acceptable: " << tensorShapeToString(requestInputItr->shape()) << "; input name: " << requestedName;
                const std::string details = ss.str();
                SPDLOG_DEBUG("[servable name: {} version: {}] Invalid shape - {}", request.model_name(), request.model_version(), details);
                return Status(StatusCode::INVALID_SHAPE, details);
            }
            shape.push_back(requestInputItr->shape()[i]);
        }
        ov::element::Type precision = ovmsPrecisionToIE2Precision(KFSPrecisionToOvmsPrecision(requestInputItr->datatype()));
        size_t expectElementsCount = ov::shape_size(shape.begin(), shape.end());
        size_t expectedBytes = precision.size() * expectElementsCount;
        if (expectedBytes <= 0) {
            std::stringstream ss;
            ss << "Invalid precision with expected bytes equal to 0: " << requestInputItr->datatype() << "; input name: " << requestedName;
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] {}", request.model_name(), request.model_version(), details);
            return Status(StatusCode::INVALID_PRECISION, details);
        }
        if (expectedBytes != bufferLocation.size()) {
            std::stringstream ss;
            ss << "Expected: " << expectedBytes << " bytes; Actual: " << bufferLocation.size() << " bytes; input name: " << requestedName;
            const std::string details = ss.str();
            SPDLOG_DEBUG("[servable name: {} version: {}] Invalid content size of tensor proto - {}", request.model_name(), request.model_version(), details);
            return Status(StatusCode::INVALID_CONTENT_SIZE, details);
        }
        outTensor = ov::Tensor(precision, shape, const_cast<void*>((const void*)bufferLocation.data()));
        return StatusCode::OK;
    } catch (const std::exception& e) {
        SPDLOG_DEBUG("Kserve mediapipe request deserialization failed:{}", e.what());
    } catch (...) {
        SPDLOG_DEBUG("KServe mediapipe request deserialization failed");
    }
    return Status(StatusCode::INTERNAL_ERROR, "Unexpected error during Tensor creation");
}

static Status matFormatToImageFormat(const size_t& matFormat, mediapipe::ImageFormat::Format& imageFormat) {
    switch (matFormat) {
    case CV_8UC1:
        imageFormat = mediapipe::ImageFormat::GRAY8;
        break;
    case CV_8UC3:
        imageFormat = mediapipe::ImageFormat::SRGB;
        break;
    case CV_8UC4:
        imageFormat = mediapipe::ImageFormat::SRGBA;
        break;
    case CV_16UC1:
        imageFormat = mediapipe::ImageFormat::GRAY16;
        break;
    case CV_16UC3:
        imageFormat = mediapipe::ImageFormat::SRGB48;
        break;
    case CV_16UC4:
        imageFormat = mediapipe::ImageFormat::SRGBA64;
        break;
    case CV_8SC1:
        imageFormat = mediapipe::ImageFormat::GRAY8;
        break;
    case CV_8SC3:
        imageFormat = mediapipe::ImageFormat::SRGB;
        break;
    case CV_8SC4:
        imageFormat = mediapipe::ImageFormat::SRGBA;
        break;
    case CV_16SC1:
        imageFormat = mediapipe::ImageFormat::GRAY16;
        break;
    case CV_16SC3:
        imageFormat = mediapipe::ImageFormat::SRGB48;
        break;
    case CV_16SC4:
        imageFormat = mediapipe::ImageFormat::SRGBA64;
        break;
    case CV_32FC1:
        imageFormat = mediapipe::ImageFormat::VEC32F1;
        break;
    case CV_32FC2:
        imageFormat = mediapipe::ImageFormat::VEC32F2;
        break;
    // case CV_32FC4:
    //     imageFormat = mediapipe::ImageFormat::VEC32F4;
    //     break;
    default:
        return StatusCode::INTERNAL_ERROR;
        break;
    }
    return StatusCode::OK;
}

static Status deserializeTensor(const std::string& requestedName, const KFSRequest& request, mediapipe::Image& outTensor) {
    auto requestInputItr = request.inputs().begin();
    auto status = getRequestInput(requestInputItr, requestedName, request);
    if (!status.ok()) {
        SPDLOG_ERROR("Getting Input failed");
        return status;
    }
    // TODO there is no sense to check this for every input
    if (request.raw_input_contents().size() == 0 || request.raw_input_contents().size() != request.inputs().size()) {
        std::stringstream ss;
        ss << "Cannot find data in raw_input_content for input with name: " << requestedName;
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid message structure - {}", request.model_name(), request.model_version(), details);
        return Status(StatusCode::INVALID_MESSAGE_STRUCTURE, details);
    }
    auto inputIndex = requestInputItr - request.inputs().begin();
    auto& bufferLocation = request.raw_input_contents().at(inputIndex);

    if (requestInputItr->shape().size() != 3) {
        SPDLOG_ERROR("Invalid Mediapipe Image input shape size. Expected: 3 Actual: {}", requestInputItr->shape().size());
        return Status(StatusCode::INTERNAL_ERROR, "Unexpected error during Tensor creation");
    }
    size_t matFormat = 0;
    status = convertKFSDataTypeToMatFormat(requestInputItr->datatype(), matFormat);
    if (!status.ok()) {
        SPDLOG_ERROR("Received tensor datatype: {} is not supported for MediaPipe::Image format", requestInputItr->datatype());
        return status;
    }
    size_t numberOfPixels = requestInputItr->shape()[0] * requestInputItr->shape()[1];
    size_t numberOfChannels = requestInputItr->shape()[2];
    if (numberOfChannels == 0) {
        SPDLOG_ERROR("Invalid Mediapipe Image input. Cannot calculate number of channels from input shape.");
        return Status(StatusCode::INTERNAL_ERROR, "Unexpected error during Tensor creation");
    }
    auto matFormatWithChannels = CV_MAKETYPE(matFormat, numberOfChannels);
    SPDLOG_ERROR("IMAGE matFormatWithChannels {}, Number of Channels {}", matFormatWithChannels, numberOfChannels);
    cv::Mat camera_frame(requestInputItr->shape()[0], requestInputItr->shape()[1], matFormatWithChannels);
    size_t expectedSize = numberOfPixels * numberOfChannels * camera_frame.elemSize1();
    std::memcpy(camera_frame.data, bufferLocation.data(), expectedSize);
    mediapipe::ImageFormat::Format imageFormat = mediapipe::ImageFormat::UNKNOWN;
    status = matFormatToImageFormat(matFormatWithChannels, imageFormat);
    SPDLOG_ERROR("IMAGE FORMAT {}", imageFormat);
    if (!status.ok()) {
        SPDLOG_ERROR("Invalid cv::Mat format {}. Cannot convert to MediaPipe::Format.", matFormatWithChannels);
        return status;
    }
    auto outTensorFrame = std::make_shared<mediapipe::ImageFrame>(
        imageFormat, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(outTensorFrame.get());
    camera_frame.copyTo(input_frame_mat);
    SPDLOG_ERROR("IMAGE FORMAT {}", outTensor.image_format());
    outTensor = mediapipe::Image(outTensorFrame);
    SPDLOG_ERROR("IMAGE Frame FORMAT {}", outTensorFrame->Format());
    SPDLOG_ERROR("IMAGE FORMAT {}", outTensor.GetImageFrameSharedPtr()->Format());
    SPDLOG_ERROR("IMAGE FORMAT {}", outTensor.image_format());
    SPDLOG_ERROR("Uses GPU {}", outTensor.UsesGpu());
    

    return StatusCode::OK;
}

MediapipeGraphExecutor::MediapipeGraphExecutor(const std::string& name, const std::string& version, const ::mediapipe::CalculatorGraphConfig& config,
    stream_types_mapping_t inputTypes,
    stream_types_mapping_t outputTypes,
    std::vector<std::string> inputNames, std::vector<std::string> outputNames) :
    name(name),
    version(version),
    config(config),
    inputTypes(std::move(inputTypes)),
    outputTypes(std::move(outputTypes)),
    inputNames(std::move(inputNames)),
    outputNames(std::move(outputNames)) {}

namespace {
enum : unsigned int {
    INITIALIZE_GRAPH,
    RUN_GRAPH,
    ADD_INPUT_PACKET,
    FETCH_OUTPUT,
    ALL_FETCH,
    TOTAL,
    TIMER_END
};
}  // namespace

static std::map<std::string, mediapipe::Packet> createInputSidePackets(const KFSRequest* request) {
    std::map<std::string, mediapipe::Packet> inputSidePackets;
    for (const auto& [name, valueChoice] : request->parameters()) {
        if (valueChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kStringParam) {
            inputSidePackets[name] = mediapipe::MakePacket<std::string>(valueChoice.string_param()).At(mediapipe::Timestamp(0));  // TODO timestamp of side packets
        } else if (valueChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kInt64Param) {
            inputSidePackets[name] = mediapipe::MakePacket<int64_t>(valueChoice.int64_param()).At(mediapipe::Timestamp(0));  // TODO timestamp of side packets
        } else if (valueChoice.parameter_choice_case() == inference::InferParameter::ParameterChoiceCase::kBoolParam) {
            inputSidePackets[name] = mediapipe::MakePacket<bool>(valueChoice.bool_param()).At(mediapipe::Timestamp(0));  // TODO timestamp of side packets
        } else {
            SPDLOG_DEBUG("Handling parameters of different types than: bool, string, int64 is not supported");
        }
    }
    return inputSidePackets;
}

template <typename T>
static Status createPacketAndPushIntoGraph(const std::string& name, const KFSRequest& request, ::mediapipe::CalculatorGraph& graph) {
    if (name.empty()) {
        SPDLOG_DEBUG("Creating Mediapipe graph inputs name failed for: {}", name);
        return StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM;
    }
    SPDLOG_DEBUG("Tensor to deserialize:\"{}\"", name);
    T input_tensor;
    auto status = deserializeTensor(name, request, input_tensor);
    if (!status.ok()) {
        SPDLOG_DEBUG("Failed to deserialize tensor: {}", name);
        return status;
    }
    auto absStatus = graph.AddPacketToInputStream(
        name, ::mediapipe::MakePacket<T>(std::move(input_tensor)).At(::mediapipe::Timestamp(0)));
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("Failed to add stream: {} packet to mediapipe graph: {} with error: {}",
            name, request.model_name(), absStatus.message(), absStatus.raw_code());
        return Status(StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM, std::move(absMessage));
    }
    absStatus = graph.CloseInputStream(name);
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("Failed to close stream: {} of mediapipe graph: {} with error: {}",
            name, request.model_name(), absMessage);
        return Status(StatusCode::MEDIAPIPE_GRAPH_CLOSE_INPUT_STREAM_ERROR, std::move(absMessage));
    }
    return StatusCode::OK;
}

template <>
Status createPacketAndPushIntoGraph<KFSRequest*>(const std::string& name, const KFSRequest& request, ::mediapipe::CalculatorGraph& graph) {
    if (name.empty()) {
        SPDLOG_DEBUG("Creating Mediapipe graph inputs name failed for: {}", name);
        return StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM;
    }
    SPDLOG_DEBUG("Request to passthrough:\"{}\"", name);
    auto absStatus = graph.AddPacketToInputStream(
        name, ::mediapipe::MakePacket<const KFSRequest*>(&request).At(::mediapipe::Timestamp(0)));
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("Failed to add stream: {} packet to mediapipe graph: {} with error: {}",
            name, request.model_name(), absStatus.message(), absStatus.raw_code());
        return Status(StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM, std::move(absMessage));
    }
    absStatus = graph.CloseInputStream(name);
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("Failed to close stream: {} of mediapipe graph: {} with error: {}",
            name, request.model_name(), absMessage);
        return Status(StatusCode::MEDIAPIPE_GRAPH_CLOSE_INPUT_STREAM_ERROR, std::move(absMessage));
    }
    return StatusCode::OK;
}

template <typename T>
static Status receiveAndSerializePacket(::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName);

template <>
Status receiveAndSerializePacket<tensorflow::Tensor>(::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName) {
    try {
        auto received = packet.Get<tensorflow::Tensor>();
        auto* output = response.add_outputs();
        output->set_name(outputStreamName);
        output->set_datatype(
            ovmsPrecisionToKFSPrecision(
                TFSPrecisionToOvmsPrecision(
                    received.dtype())));
        output->clear_shape();
        for (const auto& dim : received.shape()) {
            output->add_shape(dim.size);
        }
        response.add_raw_output_contents()->assign(reinterpret_cast<char*>(received.data()), received.TotalBytes());
        return StatusCode::OK;
    } catch (const std::exception& e) {
        std::stringstream ss;
        ss << "Failed to get packet"
           << outputStreamName
           << " with exception: "
           << e.what();
        std::string details{ss.str()};
        SPDLOG_DEBUG(details);
        return Status(StatusCode::UNKNOWN_ERROR, std::move(details));
    }
}

template <>
Status receiveAndSerializePacket<ov::Tensor>(::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName) {
    try {
        auto received = packet.Get<ov::Tensor>();
        auto* output = response.add_outputs();
        output->set_name(outputStreamName);
        output->set_datatype(
            ovmsPrecisionToKFSPrecision(
                ovElementTypeToOvmsPrecision(
                    received.get_element_type())));
        output->clear_shape();
        for (const auto& dim : received.get_shape()) {
            output->add_shape(dim);
        }
        response.add_raw_output_contents()->assign(reinterpret_cast<char*>(received.data()), received.get_byte_size());
        return StatusCode::OK;
    } catch (const std::exception& e) {
        std::stringstream ss;
        ss << "Failed to get packet"
           << outputStreamName
           << " with exception: "
           << e.what();
        std::string details{ss.str()};
        SPDLOG_DEBUG(details);
        return Status(StatusCode::UNKNOWN_ERROR, std::move(details));
    }
}

static Status convertImageFormatToKFSDataType(const mediapipe::ImageFormat::Format& imageFormat, KFSDataType& datatype) {
    switch (imageFormat) {
    case mediapipe::ImageFormat::GRAY8:
        datatype = "UINT8";
        break;
    case mediapipe::ImageFormat::SRGB:
        datatype = "UINT8";
        break;
    case mediapipe::ImageFormat::SRGBA:
        datatype = "UINT8";
        break;
    case mediapipe::ImageFormat::GRAY16:
        datatype = "UINT8";
        break;
    case mediapipe::ImageFormat::SRGB48:
        datatype = "UINT16";
        break;
    case mediapipe::ImageFormat::SRGBA64:
        datatype = "UINT16";
        break;
    case mediapipe::ImageFormat::VEC32F1:
        datatype = "FP32";
        break;
    case mediapipe::ImageFormat::VEC32F2:
        datatype = "FP32";
        break;
    // case CV_32FC4:
    //     //imageFormat = mediapipe::ImageFormat::VEC32F4;
    //     break;
    default:
        return StatusCode::INTERNAL_ERROR;
        break;
    }
    return StatusCode::OK;
}

static int GetMatType(const mediapipe::ImageFormat::Format format) {
  int type = 0;
  switch (format) {
    case mediapipe::ImageFormat::UNKNOWN:
      // Invalid; Default to uchar.
      type = CV_8U;
      break;
    case mediapipe::ImageFormat::SRGB:
      type = CV_8U;
      break;
    case mediapipe::ImageFormat::SRGBA:
      type = CV_8U;
      break;
    case mediapipe::ImageFormat::GRAY8:
      type = CV_8U;
      break;
    case mediapipe::ImageFormat::GRAY16:
      type = CV_16U;
      break;
    case mediapipe::ImageFormat::YCBCR420P:
      // Invalid; Default to uchar.
      type = CV_8U;
      break;
    case mediapipe::ImageFormat::YCBCR420P10:
      // Invalid; Default to uint16.
      type = CV_16U;
      break;
    case mediapipe::ImageFormat::SRGB48:
      type = CV_16U;
      break;
    case mediapipe::ImageFormat::SRGBA64:
      type = CV_16U;
      break;
    case mediapipe::ImageFormat::VEC32F1:
      type = CV_32F;
      break;
    case mediapipe::ImageFormat::VEC32F2:
      type = CV_32FC2;
      break;
    // case mediapipe::ImageFormat::VEC32F4:
    //   type = CV_32FC4;
    //   break;
    case mediapipe::ImageFormat::LAB8:
      type = CV_8U;
      break;
    case mediapipe::ImageFormat::SBGRA:
      type = CV_8U;
      break;
    default:
      type = CV_8U;
      break;
  }
  return type;
}

template <>
Status receiveAndSerializePacket<mediapipe::Image>(::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName) {
    const mediapipe::Image received = packet.Get<mediapipe::Image>();
    auto* output = response.add_outputs();
    output->set_name(outputStreamName);
    KFSDataType datatype;
    auto status = convertImageFormatToKFSDataType(received.GetImageFrameSharedPtr()->Format(), datatype);
    if (!status.ok()){
        SPDLOG_DEBUG("Output mediapipe::ImageFormat {} conversion to KFS Datatype failed.", received.image_format());
        return status;
    }
    output->set_datatype(datatype);
    output->clear_shape();
    output->add_shape(received.GetImageFrameSharedPtr()->Height());
    output->add_shape(received.GetImageFrameSharedPtr()->Width());
    output->add_shape(received.GetImageFrameSharedPtr()->ChannelSize());
    
    cv::Mat imageMat = mediapipe::formats::MatView(received.GetImageFrameSharedPtr().get());

    cv::Mat image;
    imageMat.convertTo(image, GetMatType(received.GetImageFrameSharedPtr()->Format()));

    response.add_raw_output_contents()->assign(reinterpret_cast<char*>(image.data), image.cols * image.rows * image.channels() * image.elemSize1());
    return StatusCode::OK;
}

template <>
Status receiveAndSerializePacket<KFSResponse*>(::mediapipe::Packet& packet, KFSResponse& response, const std::string& outputStreamName) {
    try {
        auto received = packet.Get<KFSResponse*>();
        if (received == nullptr) {
            std::stringstream ss;
            ss << "Received nullptr KFSResponse for: "
               << outputStreamName;
            std::string details{ss.str()};
            SPDLOG_DEBUG(details);
            return Status(StatusCode::UNKNOWN_ERROR, std::move(details));
        }
        response = std::move(*received);
    } catch (const std::exception& e) {
        std::stringstream ss;
        ss << "Failed to get packet"
           << outputStreamName
           << " with exception: "
           << e.what();
        std::string details{ss.str()};
        SPDLOG_DEBUG(details);
        return Status(StatusCode::UNKNOWN_ERROR, std::move(details));
    }
    return StatusCode::OK;
}

Status MediapipeGraphExecutor::infer(const KFSRequest* request, KFSResponse* response, ExecutionContext executionContext, ServableMetricReporter*& reporterOut) const {
    Timer<TIMER_END> timer;
    SPDLOG_DEBUG("Start KServe request mediapipe graph: {} execution", request->model_name());
    ::mediapipe::CalculatorGraph graph;
    auto absStatus = graph.Initialize(this->config);
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("KServe request for mediapipe graph: {} initialization failed with message: {}", request->model_name(), absMessage);
        return Status(StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR, std::move(absMessage));
    }

    std::unordered_map<std::string, ::mediapipe::OutputStreamPoller> outputPollers;
    for (auto& name : this->outputNames) {
        if (name.empty()) {
            SPDLOG_DEBUG("Creating Mediapipe graph outputs name failed for: {}", name);
            return StatusCode::MEDIAPIPE_GRAPH_ADD_OUTPUT_STREAM_ERROR;
        }
        auto absStatusOrPoller = graph.AddOutputStreamPoller(name);
        if (!absStatusOrPoller.ok()) {
            const std::string absMessage = absStatusOrPoller.status().ToString();
            SPDLOG_DEBUG("Failed to add mediapipe graph output stream poller: {} with error: {}", request->model_name(), absMessage);
            return Status(StatusCode::MEDIAPIPE_GRAPH_ADD_OUTPUT_STREAM_ERROR, std::move(absMessage));
        }
        outputPollers.emplace(name, std::move(absStatusOrPoller).value());
    }

    std::map<std::string, mediapipe::Packet> inputSidePackets{createInputSidePackets(request)};
    absStatus = graph.StartRun(inputSidePackets);
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("Failed to start mediapipe graph: {} with error: {}", request->model_name(), absMessage);
        return Status(StatusCode::MEDIAPIPE_GRAPH_START_ERROR, std::move(absMessage));
    }
    if (static_cast<int>(this->inputNames.size()) != request->inputs().size()) {
        std::stringstream ss;
        ss << "Expected: " << this->inputNames.size() << "; Actual: " << request->inputs().size();
        const std::string details = ss.str();
        SPDLOG_DEBUG("[servable name: {} version: {}] Invalid number of inputs - {}", request->model_name(), version, details);
        return Status(StatusCode::INVALID_NO_OF_INPUTS, details);
    }

    ::mediapipe::Packet packet;
    std::set<std::string> outputPollersWithReceivedPacket;

    // Passing whole KFS request and response
    // TODO ensure in config validation that we only allow REQUEST or TENSOR or IMAGE
    // not mix REQUEST with other types
    ovms::Status status;
    for (auto& name : this->inputNames) {
        if (this->inputTypes.at(name) == mediapipe_packet_type_enum::KFS_REQUEST) {
            SPDLOG_DEBUG("Request processing KFS passthrough: {}", name);
            status = createPacketAndPushIntoGraph<KFSRequest*>(name, *request, graph);
        } else if (this->inputTypes.at(name) == mediapipe_packet_type_enum::TFTENSOR) {
            SPDLOG_DEBUG("Request processing TF tensor: {}", name);
            status = createPacketAndPushIntoGraph<tensorflow::Tensor>(name, *request, graph);
        } else if (this->inputTypes.at(name) == mediapipe_packet_type_enum::MEDIAPIPE_IMAGE) {
            SPDLOG_DEBUG("Request processing  : {}", name);
            status = createPacketAndPushIntoGraph<mediapipe::Image>(name, *request, graph);
        } else if ((this->inputTypes.at(name) == mediapipe_packet_type_enum::OVTENSOR) ||
                   (this->inputTypes.at(name) == mediapipe_packet_type_enum::UNKNOWN)) {
            SPDLOG_DEBUG("Request processing OVTensor: {}", name);
            status = createPacketAndPushIntoGraph<ov::Tensor>(name, *request, graph);
        }
        if (!status.ok()) {
            return status;
        }
    }
    // receive outputs
    for (auto& [outputStreamName, poller] : outputPollers) {
        size_t receivedOutputs = 0;
        SPDLOG_DEBUG("Will wait for output stream: {} packet", outputStreamName);
        // Size checked to be equal 1 at the beggining of the function
        while (poller.Next(&packet)) {
            SPDLOG_DEBUG("Received packet from output stream: {}", outputStreamName);
            if (this->outputTypes.at(outputStreamName) == mediapipe_packet_type_enum::KFS_RESPONSE) {
                SPDLOG_DEBUG("Response processing packet type KFSPass name: {}", outputStreamName);
                status = receiveAndSerializePacket<KFSResponse*>(packet, *response, outputStreamName);
            } else if (this->outputTypes.at(outputStreamName) == mediapipe_packet_type_enum::TFTENSOR) {
                SPDLOG_DEBUG("Response processing packet type TF Tensor name: {}", outputStreamName);
                status = receiveAndSerializePacket<tensorflow::Tensor>(packet, *response, outputStreamName);
            } else if (this->outputTypes.at(outputStreamName) == mediapipe_packet_type_enum::MEDIAPIPE_IMAGE) {
                SPDLOG_DEBUG("Response processing Mediapipe Image: {}", outputStreamName);
                status = receiveAndSerializePacket<mediapipe::Image>(packet, *response, outputStreamName);
            } else if ((this->outputTypes.at(outputStreamName) == mediapipe_packet_type_enum::OVTENSOR) ||
                       (this->outputTypes.at(outputStreamName) == mediapipe_packet_type_enum::UNKNOWN)) {
                SPDLOG_DEBUG("Response processing packet type:  OVTensor name: {}", outputStreamName);
                status = receiveAndSerializePacket<ov::Tensor>(packet, *response, outputStreamName);
            }
            if (!status.ok()) {
                return status;
            }
            outputPollersWithReceivedPacket.insert(outputStreamName);
            ++receivedOutputs;
        }
        SPDLOG_TRACE("Received all: {} packets for: {}", receivedOutputs, outputStreamName);
    }
    absStatus = graph.WaitUntilDone();
    if (!absStatus.ok()) {
        const std::string absMessage = absStatus.ToString();
        SPDLOG_DEBUG("Mediapipe failed to execute: {}", absMessage);
        return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, absMessage);
    }
    if (outputPollers.size() != outputPollersWithReceivedPacket.size()) {
        SPDLOG_DEBUG("Mediapipe failed to execute. Failed to receive all output packets");
        return Status(StatusCode::MEDIAPIPE_EXECUTION_ERROR, "Unknown error during mediapipe execution");
    }
    SPDLOG_DEBUG("Received all output stream packets for graph: {}", request->model_name());
    response->set_model_name(request->model_name());
    response->set_id(request->id());
    response->set_model_version(request->model_version());
    return StatusCode::OK;
}

}  // namespace ovms
