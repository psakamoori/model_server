#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <unistd.h>
#include <memory>

#include "custom_node_interface.h"

#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>

using namespace boost::interprocess;

// Define an STL compatible allocator of ints that allocates from the managed_shared_memory.
// This allocator will allow placing containers in the segment
typedef allocator<uint8_t, managed_shared_memory::segment_manager> ShmemAllocator;

// Alias a vector that uses the previous STL-like allocator so that allocates
// its values from the segment
typedef vector<uint8_t, ShmemAllocator> MyVector;

class MyManager {
    const char* shm_id = "MySharedMemory";
    bool __cleaned;
    const size_t max_mem = 1024 * 1024 * 1024;  // 1gb
    managed_shared_memory segment;
    const ShmemAllocator alloc_inst;
public:
    std::unique_ptr<named_semaphore> sem1;
    std::unique_ptr<named_semaphore> sem2;
    MyManager() :
        __cleaned(shared_memory_object::remove(shm_id)),
        segment(managed_shared_memory(create_only, shm_id, max_mem)),
        alloc_inst(segment.get_segment_manager())
    {
        named_semaphore::remove("MySem1");
        named_semaphore::remove("MySem2");
        sem1 = std::make_unique<named_semaphore>(open_or_create, "MySem1", 0);
        sem2 = std::make_unique<named_semaphore>(open_or_create, "MySem2", 0);
        std::cout << "------------- Hi!" << std::endl;
    }
    ~MyManager() {
        named_semaphore::remove("MySem1");
        named_semaphore::remove("MySem2");
        shared_memory_object::remove(shm_id);
        std::cout << "------------- Bye!" << std::endl;
    }

    managed_shared_memory& getSegment() { return this->segment; }
    const ShmemAllocator& getAllocator() const { return this->alloc_inst; }
};

class measure {
    std::chrono::steady_clock::time_point now;
    std::string msg;
public:
    measure(const std::string& msg) : msg(msg) {
        now = std::chrono::steady_clock::now();
    }
    ~measure() {
        auto end = std::chrono::steady_clock::now();
        std::cout << msg << std::chrono::duration_cast<std::chrono::microseconds>(end - now).count() / 1000.0 << "ms" << std::endl;
    }
};

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount) {
    *customNodeLibraryInternalManager = (void*) new MyManager();

    if (fork() == 0) {
        // child
        char* command = "bazel-bin/src/processing_worker";
        char* argument_list[] = {"bazel-bin/src/processing_worker", NULL};

        int status_code = execvp(command, argument_list);
        if (status_code == -1) {
            printf("Terminated Incorrectly\n");
        } else {
            printf("Thread executed!\n");
        }
        fflush(stdout);
        exit(0);
    } else {
        // parent

    }

    return 0;
}


int deinitialize(void* customNodeLibraryInternalManager){
    delete (MyManager*)(customNodeLibraryInternalManager);
    return 0;
}

int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    auto* mgr = (MyManager*)(customNodeLibraryInternalManager);
    if (inputsCount != 1) {
        return 2;
    }

    if (strcmp(inputs[0].name, "in") != 0) {
        return 3;
    }

    const struct CustomNodeTensor* input = &inputs[0];

    if (input->precision != U8) {
        return 4;
    }

    std::string id;
    MyVector* input_shm;
    try {
    {
        measure m("------- Creating object id time: ");
        //id = std::to_string((int64_t)input->data);
        id = "MyVector";
    } {
        measure m("------- Constructing vector time: ");
        input_shm = mgr->getSegment().find_or_construct<MyVector>(id.c_str())(mgr->getAllocator());
    } {
        measure m("------- Resize vector time: ");
        input_shm->resize(input->dataBytes);
    } {
        measure m("------- Copying tensor to shared memory vector: ");
        std::memcpy(input_shm->data(), input->data, input->dataBytes);
    } {
        measure m("------- Editing shared memory in another process(+1): ");
        mgr->sem1->post();
        mgr->sem2->wait();
    }
    /* 120mb

------- Creating object id time: 0ms
------- Constructing vector time: 0.001ms
------- Resize vector time: 0ms
------- Copying tensor to shared memory vector: 13.215ms
Processing.... --------------------
Finished processing! --------------------
------- Editing shared memory in another process(+1): 201.503ms
------- Copying from shared memory vector to output tensor: 92.841ms
------- Removing boost managed vector: 0ms

    */

    *outputsCount = 1;
    *outputs = (struct CustomNodeTensor*)malloc(sizeof(struct CustomNodeTensor) * (*outputsCount));
    struct CustomNodeTensor* output = (&(*outputs))[0];

    output->name = "out";
    output->data = (uint8_t*)malloc(input->dataBytes * sizeof(uint8_t));
    output->dataBytes = input->dataBytes;
    output->dims = (uint64_t*)malloc(input->dimsCount * sizeof(uint64_t));
    output->dimsCount = input->dimsCount;
    memcpy((void*)output->dims, (void*)input->dims, input->dimsCount * sizeof(uint64_t));
    output->precision = input->precision;

    {
        measure m("------- Copying from shared memory vector to output tensor: ");
        std::memcpy(output->data, input_shm->data(), output->dataBytes);
    } {
        measure m("------- Removing boost managed vector: ");
        // mgr->getSegment().destroy<MyVector>(id.c_str());
    }
    } catch(interprocess_exception &ex){
      std::cout << ex.what() << std::endl;
      return 1;
   }
    return 0;
}

int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*) malloc (*infoCount * sizeof(struct CustomNodeTensorInfo));
    (*info)->name = "in";
    (*info)->dimsCount = 4;
    (*info)->dims = (uint64_t*) malloc((*info)->dimsCount * sizeof(uint64_t));
    (*info)->dims[0] = 40;
    (*info)->dims[1] = 3;
    (*info)->dims[2] = 1024;
    (*info)->dims[3] = 1024;  // 120mb
    (*info)->precision = U8;
    return 0;
}

int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager) {
    *infoCount = 1;
    *info = (struct CustomNodeTensorInfo*) malloc (*infoCount * sizeof(struct CustomNodeTensorInfo));
    (*info)->name = "out";
    (*info)->dimsCount = 4;
    (*info)->dims = (uint64_t*) malloc((*info)->dimsCount * sizeof(uint64_t));
    (*info)->dims[0] = 40;
    (*info)->dims[1] = 3;
    (*info)->dims[2] = 1024;
    (*info)->dims[3] = 1024;  // 120mb
    (*info)->precision = U8;
    return 0;
}

int release(void* ptr, void* customNodeLibraryInternalManager) {
    free(ptr);
    return 0;
}
