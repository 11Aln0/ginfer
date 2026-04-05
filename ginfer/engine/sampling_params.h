#pragma once

namespace ginfer::engine {

struct SamplingParams {
  int max_tokens = 64;
  bool ignore_eos = false;
};

}  // namespace ginfer::engine