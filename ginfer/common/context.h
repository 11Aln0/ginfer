#pragma once

#include <optional>

namespace ginfer::common {

struct InferContext {
  std::optional<int> max_seqlen_q;

  InferContext() = default;

  InferContext& setMaxSeqlenQ(int max_seqlen_q) {
    this->max_seqlen_q = max_seqlen_q;
    return *this;
  };
};

}  // namespace ginfer::common