#include <iostream>
#include <vector>
#include <algorithm>

#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>

using namespace boost::interprocess;

typedef allocator<uint8_t, managed_shared_memory::segment_manager> ShmemAllocator;
typedef vector<uint8_t, ShmemAllocator> MyVector;

int main() {
    std::cout << "int main() start of processing_worker ---------------------" << std::endl;

    // Open "MySharedMemory" region (1gb defined in custom node)
    managed_shared_memory segment(open_only, "MySharedMemory");
    const ShmemAllocator alloc_inst(segment.get_segment_manager());

    // Find reference to vector which is used to pass data CN <=> processing_worker
    MyVector* input_shm = segment.find_or_construct<MyVector>("MyVector")(alloc_inst);
    if (!input_shm) {
        std::cout << "Error!" << std::endl;
        return 1;
    }

    named_semaphore sem1(open_or_create, "MySem1", 0);  // semaphore for waiting for processing request
    named_semaphore sem2(open_or_create, "MySem2", 0);  // semaphore for signaling response ready

    int iterations = 5;
    while (iterations-- > 0) {
        sem1.wait();  // wait for processing request
        std::cout << "Processing.... --------------------" << std::endl;

        // add 1 to each byte of the request
        std::for_each(input_shm->begin(), input_shm->end(), [](uint8_t& e) {
                e += 1;
            });
        std::cout << "Finished processing! --------------------" << std::endl;

        sem2.post();  // inform custom node that response is ready
    }

    std::cout << "int main() end of processing_worker ------------------------" << std::endl;
    return 0;
}
