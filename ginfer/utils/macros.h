#pragma once

namespace ginfer::utils {

#define FOR_EACH_1(WHAT, X) WHAT(X)
#define FOR_EACH_2(WHAT, X, ...) WHAT(X) FOR_EACH_1(WHAT, __VA_ARGS__)
#define FOR_EACH_3(WHAT, X, ...) WHAT(X) FOR_EACH_2(WHAT, __VA_ARGS__)
#define FOR_EACH_4(WHAT, X, ...) WHAT(X) FOR_EACH_3(WHAT, __VA_ARGS__)
#define FOR_EACH_5(WHAT, X, ...) WHAT(X) FOR_EACH_4(WHAT, __VA_ARGS__)
#define FOR_EACH_6(WHAT, X, ...) WHAT(X) FOR_EACH_5(WHAT, __VA_ARGS__)
#define FOR_EACH_7(WHAT, X, ...) WHAT(X) FOR_EACH_6(WHAT, __VA_ARGS__)
#define FOR_EACH_8(WHAT, X, ...) WHAT(X) FOR_EACH_7(WHAT, __VA_ARGS__)
#define FOR_EACH_9(WHAT, X, ...) WHAT(X) FOR_EACH_8(WHAT, __VA_ARGS__)
#define FOR_EACH_10(WHAT, X, ...) WHAT(X) FOR_EACH_9(WHAT, __VA_ARGS__)

#define GET_FOR_EACH_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, NAME, ...) NAME
#define FOR_EACH(WHAT, ...)                                                                    \
  GET_FOR_EACH_MACRO(__VA_ARGS__, FOR_EACH_10, FOR_EACH_9, FOR_EACH_8, FOR_EACH_7, FOR_EACH_6, \
                     FOR_EACH_5, FOR_EACH_4, FOR_EACH_3, FOR_EACH_2, FOR_EACH_1)               \
  (WHAT, __VA_ARGS__)

}  // namespace ginfer::utils