#include "metal.hh"
#include "runtime.hh"
#include "story/tensor.hh"

namespace minimlx {
Tensor add_cpu(Runtime &r, const Tensor &a, const Tensor &b);
Tensor add(Runtime &r, const Tensor &a, const Tensor &b,
           bool with_relu = false);

Tensor add(Runtime &r, const Tensor &a, const float b, bool with_relu = false);

Tensor pack(Runtime &r, const Tensor &x, bool force_copy = false);

Tensor multiply(Runtime &r, const Tensor &a, const float b,
                bool with_relu = false);
} // namespace minimlx
