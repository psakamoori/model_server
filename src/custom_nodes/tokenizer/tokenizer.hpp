#include <string>
#include <cstdint>
#include <vector>

#include "../../custom_node_interface.h"

namespace custom_nodes {
namespace tokenizer {

class Model {
    int id;
    void* handle = nullptr;

public:
    Model(const std::string& modelPath);
    ~Model();

    const int tokenize(const std::string& text, int32_t* ids, int maxIdsArrLength);
    std::vector<int64_t> tokenizeEx(const std::string& text, int maxIdsArrLength);
};

}  // namespace tokenizer
}  // namespace custom_nodes

int initialize(void** customNodeLibraryInternalManager, const struct CustomNodeParam* params, int paramsCount);
int deinitialize(void* customNodeLibraryInternalManager);
int execute(const struct CustomNodeTensor* inputs, int inputsCount, struct CustomNodeTensor** outputs, int* outputsCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager);
int getInputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager);
int getOutputsInfo(struct CustomNodeTensorInfo** info, int* infoCount, const struct CustomNodeParam* params, int paramsCount, void* customNodeLibraryInternalManager);
int release(void* ptr, void* customNodeLibraryInternalManager);
