#include "metal.hh"
#include "story/tensor.hh"

namespace minimlx {
Tensor add_cpu(MTL::Device *device, const Tensor &a, const Tensor &b);
}
