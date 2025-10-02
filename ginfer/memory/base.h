#pragma once

namespace ginfer::memory {

class NoCopyable {
 protected:
  NoCopyable() = default;  // deleted Copy constructor should assign a default constructor
  ~NoCopyable() = default;
  NoCopyable(const NoCopyable&) = delete;
  NoCopyable& operator=(const NoCopyable&) = delete;
};

}  // namespace ginfer::memory