#include "model_loader.h"
#include <fstream>
#include <fmt/core.h>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

model_loader::model_loader(const std::string& fname) {
    
    std::ifstream fin(fname, std::ios::binary);

    if(!fin) {
        throw std::runtime_error(fmt::format("Failed to open file {}", fname));
    }

    uint64_t header_len;
    fin.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));

    std::vector<char> header_buf(header_len);
    fin.read(header_buf.data(), header_len);

    json header = json::parse(header_buf);

    
}
