#pragma once

#include <fstream>
#include <string>
#include "ginfer/common/errors.h"

namespace ginfer::utils::file {

Result<std::string, std::string> loadBytesFromFile(const std::string& path);

}  // namespace ginfer::utils::file