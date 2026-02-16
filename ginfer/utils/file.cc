
#include "ginfer/utils/utils.h"

namespace ginfer::utils::file {

Result<std::string, std::string> loadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  RETURN_ERR_ON(fs.fail(), "Cannot open file: {}", path);
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return Ok(std::move(data));
}

}  // namespace ginfer::utils::file