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
#include "capi_utils.hpp"

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "../logging.hpp"
#include "../precision.hpp"
#include "../profiler.hpp"
#include "../pocapi.hpp"
#include "../status.hpp"

namespace ovms {
Precision PrecisionToOvmsPrecision(const OVMSDataType& datatype) {
    static std::unordered_map<OVMSDataType, Precision> precisionMap{
        {OVMS_DATATYPE_BOOL, Precision::BOOL},
        {OVMS_DATATYPE_FP64, Precision::FP64},
        {OVMS_DATATYPE_FP32, Precision::FP32},
        {OVMS_DATATYPE_FP16, Precision::FP16},
        {OVMS_DATATYPE_I64, Precision::I64},
        {OVMS_DATATYPE_I32, Precision::I32},
        {OVMS_DATATYPE_I16, Precision::I16},
        {OVMS_DATATYPE_I8, Precision::I8},
        {OVMS_DATATYPE_U64, Precision::U64},
        {OVMS_DATATYPE_U32, Precision::U32},
        {OVMS_DATATYPE_U16, Precision::U16},
        // {"BYTES", Precision::??},
        {OVMS_DATATYPE_U8, Precision::U8}};
    auto it = precisionMap.find(datatype);
    if (it == precisionMap.end()) {
        return Precision::UNDEFINED;
    }
    return it->second;
}

size_t DataTypeSize(const OVMSDataType& datatype) {
    static std::unordered_map<OVMSDataType, size_t> datatypeSizeMap{
        {OVMS_DATATYPE_BOOL, 1},
        {OVMS_DATATYPE_U8, 1},
        {OVMS_DATATYPE_U16, 2},
        {OVMS_DATATYPE_U32, 4},
        {OVMS_DATATYPE_U64, 8},
        {OVMS_DATATYPE_I8, 1},
        {OVMS_DATATYPE_I16, 2},
        {OVMS_DATATYPE_I32, 4},
        {OVMS_DATATYPE_I64, 8},
        {OVMS_DATATYPE_FP16, 2},
        {OVMS_DATATYPE_FP32, 4},
        {OVMS_DATATYPE_FP64, 8}
        // {"BYTES", },
    };
    auto it = datatypeSizeMap.find(datatype);
    if (it == datatypeSizeMap.end()) {
        return 0;
    }
    return it->second;
}

const OVMSDataType& ovmsPrecisionToPrecision(Precision precision) {
    static std::unordered_map<Precision, OVMSDataType> precisionMap{
        {Precision::FP64, OVMS_DATATYPE_FP64},
        {Precision::FP32, OVMS_DATATYPE_FP32},
        {Precision::FP16, OVMS_DATATYPE_FP16},
        {Precision::I64, OVMS_DATATYPE_I64},
        {Precision::I32, OVMS_DATATYPE_I32},
        {Precision::I16, OVMS_DATATYPE_I16},
        {Precision::I8, OVMS_DATATYPE_I8},
        {Precision::U64, OVMS_DATATYPE_U64},
        {Precision::U32, OVMS_DATATYPE_U32},
        {Precision::U16, OVMS_DATATYPE_U16},
        {Precision::U8, OVMS_DATATYPE_U8},
        {Precision::BOOL, OVMS_DATATYPE_BOOL},
        {Precision::INVALID, OVMS_DATATYPE_INVALID}};
        
    // {Precision::BF16, ""},
    // {Precision::U4, ""},
    // {Precision::U1, ""},
    // {Precision::CUSTOM, ""},
    // {Precision::DYNAMIC, ""},
    // {Precision::MIXED, ""},
    // {Precision::Q78, ""},
    // {Precision::BIN, ""},
    // {Precision::I4, ""},
    // {Precision::UNDEFINED, "UNDEFINED"}};
    
    auto it = precisionMap.find(precision);
    if (it == precisionMap.end()) {
        it = precisionMap.find(Precision::INVALID);
    }
    return it->second;
}

std::string tensorShapeToString(const Shape& shape) {
    std::ostringstream oss;
    oss << "(";
    size_t i = 0;
    if (shape.size() > 0) {
        for (; i < shape.size() - 1; i++) {
            oss << shape[i].toString() << ",";
        }
        oss << shape[i].toString();
    }
    oss << ")";

    return oss.str();
}

Status prepareConsolidatedTensorImpl(InferenceResponse* response, char*& bufferOut, const std::string& name, size_t size) {
    OVMS_PROFILE_FUNCTION();
    // TODO:
    //for (int i = 0; i < response->outputs_size(); i++) {
    //    if (response->mutable_outputs(i)->name() == name) {
    //        SPDLOG_LOGGER_ERROR(dag_executor_logger, "Failed to prepare consolidated tensor, tensor with name {} already prepared", name);
    //        return StatusCode::INTERNAL_ERROR;
    //    }
    //}
    
    //auto* proto = response->add_outputs();
    //proto->set_name(name);
    //auto* content = response->add_raw_output_contents();
    //content->resize(size);
    //bufferOut = content->data();
    return StatusCode::OK;
}
}  // namespace ovms
