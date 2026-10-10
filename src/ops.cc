#include "story/ops.hh"

namespace minimlx {

Tensor add_cpu(MTL::Device *device, const Tensor &a, const Tensor &b) {

  if (a.shape() != b.shape())
    throw std::invalid_argument(
        "add: tensors must have the same number of elements");

  Tensor result(device, a.shape());

  for (int i = 0; i < a.numel(); ++i) {
    result[i] = a[i] + b[i];
  }
  return result;
};

} // namespace minimlx
