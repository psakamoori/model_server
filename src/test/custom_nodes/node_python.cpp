//*****************************************************************************
// Copyright 2021 Intel Corporation
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
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cstring>

#include "../../custom_node_interface.h"

#include <Python.h>
#include "numpy/arrayobject.h"


static constexpr const char* INPUT_TENSOR_NAME = "input";
static constexpr const char* OUTPUT_TENSOR_NAME = "output";

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    // Py_Initialize();
    // import_array();
    //PyRun_SimpleString("print('dupa')");
    // Py_Finalize();
    return 0;
}

int deinitialize(void* customNodeLibraryInternalManager){
    return 0;
}

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {

    Py_Initialize();
    import_array();

        // Inputs reading
    const CustomNodeTensor* inputTensor = nullptr;
    inputTensor = &(inputs[0]);

    npy_intp dims [inputTensor->dimsCount];

    for (uint64_t i = 0; i < inputTensor->dimsCount; i++) {
        dims[i] = (int32_t) inputTensor->dims[0];
    } 

    PyObject *ndarray = PyArray_SimpleNewFromData(inputTensor->dimsCount, dims, NPY_INT, reinterpret_cast<void*>(inputTensor->data));

    PyObject *dict = PyDict_New();
    PyDict_SetItemString(dict, inputTensor->name, ndarray);

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");
    PyObject *name, *load_module, *func;

    name = PyUnicode_FromString((char*)"script");

    load_module = PyImport_Import(name);

    func = PyObject_GetAttrString(load_module, (char*)"execute");
    PyObject *retDict = PyObject_CallFunctionObjArgs(func, dict, NULL);
    PyObject *retOutput = PyDict_GetItemString(retDict, "output");
    PyArrayObject* np_ret = reinterpret_cast<PyArrayObject*>(retOutput);

    uint8_t* c_out;
    c_out = reinterpret_cast<uint8_t*>(PyArray_DATA(np_ret));

    // Processing

    // Preparing output tensor
    uint64_t byteSize = sizeof(int) *  5;
    int* buffer = (int*)malloc(byteSize);

    std::memcpy((uint8_t*)buffer, c_out, byteSize);

    *outputsCount = 1;
    *outputs = (struct CustomNodeTensor*)malloc(*outputsCount * sizeof(CustomNodeTensor));
    if ((*outputs) == nullptr) {
        std::cout << "malloc has failed" << std::endl;
        free(buffer);
        return 1;
    }

    CustomNodeTensor& output = (*outputs)[0];
    output.name = "output";
    output.data = reinterpret_cast<uint8_t*>(buffer);
    output.dataBytes = byteSize;
    output.dimsCount = 1;
    output.dims = (uint64_t*)malloc(output.dimsCount * sizeof(uint64_t));
    output.dims[0] = 5;
    output.precision = I32;

    Py_Finalize();
    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));

    (*info)[0].name = "input";
    (*info)[0].dimsCount = 1;
    (*info)[0].dims = (uint64_t*)malloc((*info)[0].dimsCount * sizeof(uint64_t));
    (*info)[0].dims[0] = 5;
    (*info)[0].precision = I32;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*)malloc(*infoCount * sizeof(struct CustomNodeTensorInfo));

    (*info)[0].name = "output";
    (*info)[0].dimsCount = 1;
    (*info)[0].dims = (uint64_t*)malloc((*info)->dimsCount * sizeof(uint64_t));
    (*info)[0].dims[0] = 5;
    (*info)[0].precision = I32;

    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    free(ptr);
    return 0;
}

