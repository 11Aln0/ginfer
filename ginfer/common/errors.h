#pragma once

#include <fmt/core.h>

#include <string>

namespace ginfer::error {

enum class StatusCode {
  kSuccess = 0,
  kInvalidArgument = 1,
  kNotFound = 2,
  kNotImplemented = 3,
  kUnknownResult = 4,
};

class Status {
 public:
  explicit Status(StatusCode code, const std::string& msg) : code_(code), msg_(msg) {}

  explicit Status(StatusCode code) : code_(code), msg_("") {}

  StatusCode code() const { return code_; }

  const std::string& msg() const { return msg_; }

  bool operator==(StatusCode code) const { return code_ == code; }

 private:
  StatusCode code_ = StatusCode::kUnknownResult;
  std::string msg_ = "unknown";
};

#define REGISTER_STATUS(func, code)              \
  template <typename... Args>                    \
  Status func(Args... args) {                    \
    if constexpr (sizeof...(Args) == 0) {        \
      return Status(code);                       \
    } else {                                     \
      return Status(code, fmt::format(args...)); \
    }                                            \
  }

REGISTER_STATUS(Success, StatusCode::kSuccess)
REGISTER_STATUS(InvalidArgument, StatusCode::kInvalidArgument)
REGISTER_STATUS(NotFound, StatusCode::kNotFound)
REGISTER_STATUS(NotImplemented, StatusCode::kNotImplemented)

#define CHECK_THROW(expr, ...)                            \
  do {                                                    \
    if (!(expr)) {                                        \
      throw std::runtime_error(fmt::format(__VA_ARGS__)); \
    }                                                     \
  } while (0)

#define RETURN_ON_ERROR(status)                                  \
  do {                                                           \
    auto _status = (status);                                     \
    if (_status.code() != ginfer::error::StatusCode::kSuccess) { \
      return _status;                                            \
    }                                                            \
  } while (0)

template <typename T, typename E>
class Result {
 public:
  static Result ok(T value) { return Result(std::move(value)); }
  static Result err(E error) { return Result(std::move(error)); }

  bool hasValue() const { return has_value_; }
  const T& value() const { return value_; }
  const E& err() const { return error_; }

 private:
  Result(T&& v) : has_value_(true), value_(std::move(v)) {}
  Result(E&& e) : has_value_(false), error_(std::move(e)) {}

  bool has_value_;
  T value_;
  E error_;
};

}  // namespace ginfer::error