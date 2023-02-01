#include <iostream>
#include <string>

#include "blingfiretokdll.h"


int main() {
    const std::string model_bin = "./gpt2.bin";
    const std::string model_i2w = "./gpt2.i2w";

    const std::string input_sentence = "This is a test. Эpple pie. How do I renew my virtual smart card? ends";

    std::cout << "Loading the model..." << std::endl;
    void* h = BlingFire::LoadModel(model_bin.c_str());
    void* h_reverse = BlingFire::LoadModel(model_i2w.c_str());
    std::cout << "Loaded." << std::endl;

    int32_t ids[1024]={0,};
    int32_t expected_ids[1024] = {770, 318, 257, 1332, 13, 12466, 255, 381, 293, 2508, 13, 1374,
        466, 314, 6931, 616, 7166, 4451, 2657, 30, 5645};
    BlingFire::TextToIds(h, input_sentence.c_str(), input_sentence.size(), ids, 1024);

    std::cout << "actual:  ";
    for (int i = 0; i < 40; i++) {
        std::cout << ids[i] << " ";
    }
    std::cout << std::endl << "expected:";
    for (int i = 0; i < 40; i++) {
        std::cout << expected_ids[i] << " ";
    }
    std::cout << std::endl;

    char output_buffer[1024];
    BlingFire::IdsToText(h_reverse, ids, 21, output_buffer, 1024, false);

    std::cout << "Conversion back: " << output_buffer << std::endl;

    std::cout << "Freeing the model..." << std::endl;
    BlingFire::FreeModel(h_reverse);
    BlingFire::FreeModel(h);
    std::cout << "Freed." << std::endl;
    return 0;
}
