#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <chrono>
#include <string>

#include "custom_node_interface.h"

class MyManager {
    const char* socketPath="/ovms/src/my_socket_name";
public:
    int sockfd;
    MyManager()
    {
        sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (sockfd == -1) {
            throw std::logic_error("Failed to create socket");
        }
        struct sockaddr_un serverAddr;
        serverAddr.sun_family = AF_UNIX;
        strncpy(serverAddr.sun_path, socketPath, sizeof(serverAddr.sun_path) - 1);

        if (connect(sockfd, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
            //std::cerr << "Failed to connect to server" << std::endl;
            close(sockfd);
            throw std::logic_error("Failed to connect to server");
        }
        std::cout << "Connected to worker." << std::endl;

        std::cout << "------------- Hi!" << std::endl;
    }
    ~MyManager() {
        close(sockfd);
        std::cout << "------------- Bye!" << std::endl;
    }
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

    // if (fork() == 0) {
    //     // child
    //     char* command = "bazel-bin/src/processing_worker";
    //     char* argument_list[] = {"bazel-bin/src/processing_worker", NULL};

    //     int status_code = execvp(command, argument_list);
    //     if (status_code == -1) {
    //         printf("Terminated Incorrectly\n");
    //     } else {
    //         printf("Thread executed!\n");
    //     }
    //     fflush(stdout);
    //     exit(0);
    // } else {
    //     // parent

    // }

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

    uint32_t sizeVal;
    ssize_t sentBytes;
    {
        measure m("------- Sending 4 bytes with data length info time: ");
        sizeVal = input->dataBytes;
        sentBytes = send(mgr->sockfd, (void*)&sizeVal, 4, 0);
        if (sentBytes == -1) {
            std::cout << "Could not send sizeVal bytes" << std::endl;
            close(mgr->sockfd);
            return 1;
        }
    }

    {
        measure m("------- Sending payload time: ");
        sentBytes = send(mgr->sockfd, input->data, input->dataBytes, 0);
        if (sentBytes == -1) {
            std::cout << "Could not send bytes" << std::endl;
            close(mgr->sockfd);
            return 1;
        }
    }

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

    // {
    //     measure m("------- Copying from input tensor to output tensor: ");
    //     std::memcpy(output->data, input->data, output->dataBytes);
    // }

    ssize_t receivedBytes;
    {
        measure m("------- Receiving 4 bytes with payload length info time: ");
        receivedBytes = recv(mgr->sockfd, (void*)&sizeVal, 4, 0);
        if (receivedBytes == -1) {
            std::cout << "Count not recv sizeVal bytes" << std::endl;
            close(mgr->sockfd);
        }
        //std::cout << "received 4 bytes, sizeVal is: " << sizeVal << std::endl;
    }
    ssize_t totalRecvBytes = 0;
    {
        measure m("------- Receiving payload time: ");
        while (totalRecvBytes < sizeVal) {
            receivedBytes = recv(mgr->sockfd, output->data + totalRecvBytes, sizeVal - totalRecvBytes, 0);
            if (receivedBytes == -1 || receivedBytes == 0) {
                std::cout << "Count not recv bytes" << std::endl;
                close(mgr->sockfd);
            }
            totalRecvBytes += receivedBytes;
            // std::cout << "received bytes:  " << receivedBytes << std::endl;
        }
    }
/*
120 MB
------- Sending 4 bytes with data length info time: 0.133ms
------- Sending payload time: 302.319ms
------- Receiving 4 bytes with payload length info time: 181.696ms (181.18834495544434ms is numpy +1 addition time)
------- Receiving payload time: 114.773ms
*/

// overhead 417ms
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
