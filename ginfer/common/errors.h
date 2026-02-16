#pragma once

#include <fmt/core.h>

#include <string>
#include <variant>

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

}  // namespace ginfer::error

namespace ginfer {

template <typename T>
struct Ok {
  T value;
  Ok(T v) : value(std::move(v)) {}
};

template <typename E>
struct Err {
  E error;
  Err(E e) : error(std::move(e)) {}
};

// template <typename T>
// Ok(T) -> Ok<T>;
// template <typename E>
// Err(E) -> Err<E>;

template <typename T, typename E>
class Result {
 public:
  // static Result ok(T value) { return Result(std::move(value)); }
  // static Result err(E error) { return Result(std::move(error)); }

  Result(Ok<T> ok) : data_(std::move(ok)) {}
  Result(Err<E> err) : data_(std::move(err)) {}

  bool ok() const { return std::holds_alternative<Ok<T>>(data_); }
  const T& value() const { return std::get<Ok<T>>(data_).value; }
  const E& err() const { return std::get<Err<E>>(data_).error; }

 private:
  std::variant<Ok<T>, Err<E>> data_;
};

#define RETURN_ERR_ON(expr, ...)            \
  do {                                      \
    if ((expr)) {                           \
      return Err(fmt::format(__VA_ARGS__)); \
    }                                       \
  } while (0)

}  // namespace ginfer