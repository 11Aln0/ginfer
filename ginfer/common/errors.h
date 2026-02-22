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

}  // namespace ginfer::error

namespace ginfer {

template <typename T>
struct Ok {
  T value;

  template <typename U>
  Ok(U&& v) : value(std::forward<U>(v)) {}
};

template <>
struct Ok<void> {
  Ok() = default;
};

template <typename E>
struct Err {
  E error;

  template <typename U>
  Err(U&& e) : error(std::forward<U>(e)) {}
};

template <typename U>
Ok(U&&) -> Ok<std::decay_t<U>>;
template <typename E>
Err(E&&) -> Err<std::decay_t<E>>;

template <typename T, typename E>
class Result {
 public:
  // static Result ok(T value) { return Result(std::move(value)); }
  // static Result err(E error) { return Result(std::move(error)); }

  Result(Ok<T> ok) : data_(std::move(ok)) {}
  Result(Err<E> err) : data_(std::move(err)) {}

  bool ok() const { return std::holds_alternative<Ok<T>>(data_); }

  const E& err() const& { return std::get<Err<E>>(data_).error; }
  E&& err() && { return std::move(std::get<Err<E>>(data_).error); }
  E& err() & { return std::get<Err<E>>(data_).error; }

  template <typename U = T>
  const std::enable_if_t<!std::is_void_v<U>, U>& value() const& {
    return std::get<Ok<T>>(data_).value;
  }

  template <typename U = T>
  std::enable_if_t<!std::is_void_v<U>, U>&& value() && {
    return std::move(std::get<Ok<T>>(data_).value);
  }

  template <typename U = T>
  std::enable_if_t<!std::is_void_v<U>, U>& value() & {
    return std::get<Ok<T>>(data_).value;
  }

 private:
  std::variant<Ok<T>, Err<E>> data_;
};

#define RETURN_ERR_ON(expr, ...)            \
  do {                                      \
    if ((expr)) {                           \
      return Err(fmt::format(__VA_ARGS__)); \
    }                                       \
  } while (0)

#define DECLARE_OR_RETURN(lhs, expr)         \
  auto _res_##lhs = (expr);                  \
  if (!_res_##lhs.ok()) {                    \
    return Err(std::move(_res_##lhs).err()); \
  }                                          \
  auto lhs = std::move(_res_##lhs).value();

#define ASSIGN_OR_RETURN(lhs, expr)      \
  do {                                   \
    auto _res = (expr);                  \
    if (!_res.ok()) {                    \
      return Err(std::move(_res).err()); \
    }                                    \
    lhs = std::move(_res).value();       \
  } while (0)

#define DECLARE_OR_THROW(lhs, expr)                        \
  auto _res_##lhs = (expr);                                \
  if (!_res_##lhs.ok()) {                                  \
    throw std::runtime_error(std::move(_res_##lhs).err()); \
  }                                                        \
  auto lhs = std::move(_res_##lhs).value();

#define ASSIGN_OR_THROW(lhs, expr)                     \
  do {                                                 \
    auto _res = (expr);                                \
    if (!_res.ok()) {                                  \
      throw std::runtime_error(std::move(_res).err()); \
    }                                                  \
    lhs = std::move(_res).value();                     \
  } while (0)

#define RETURN_ON_ERR(expr)              \
  do {                                   \
    auto _res = (expr);                  \
    if (!_res.ok()) {                    \
      return Err(std::move(_res).err()); \
    }                                    \
  } while (0)

}  // namespace ginfer