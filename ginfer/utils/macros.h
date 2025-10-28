#pragma once

namespace ginfer::utils {

#define CONCAT(a, b) a##b
#define CONCAT3(a, b, c) CONCAT(CONCAT(a, b), c)


}  // namespace ginfer::utils