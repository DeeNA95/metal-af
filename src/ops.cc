#include "story/ops.hh"
#include "story/runtime.hh"
#include "story/tensor.hh"

namespace minimlx {

Tensor add_cpu(Runtime &r, const Tensor &a, const Tensor &b) {

  check(a.shape() == b.shape(), "add: tensors must have the same shape");

  Tensor result(r, a.shape());

  auto av = a.read(), bv = b.read();

  for (int i = 0; i < av.size(); ++i) {
    av[i] += bv[i];
  }
  result.assign(av);
  return result;
};

// device check
void on_device(Runtime &r, const Tensor &t) {
  check(t.buffer()->device() == r.device.get(),
        "TENSOR NOT ON SAME DEVICE AS RUNTIME");
};

// get the ideal broadcast shape
Shape broadcast_shape(Shape a, Shape b) {
  size_t rank = std::max(a.size(), b.size());

  a.insert(a.begin(), rank - a.size(), 1);
  b.insert(b.begin(), rank - b.size(), 1);

  Shape result(rank);

  for (size_t d = 0; d < rank; ++d) {
    check(a[d] == b[d] || a[d] == 1 || b[d] == 1, "broadcasting error");
    result[d] = std::max(a[d], b[d]);
  }
  elements(result);
  return result;
}

Tensor add(Runtime &r, const Tensor &a, const Tensor &b, bool with_relu) {
  on_device(r, a);
  on_device(r, b);

  Shape shape = broadcast_shape(a.shape(), b.shape());

  Tensor result(r, shape, 0); // default to zero to avoid random load overhead

  Tensor av = a.broadcast_to(shape), bv = b.broadcast_to(shape);

  AddDesc p{describe(av), describe(bv)};

  auto relu = with_relu ? "add_relu_strided" : "add_strided";

  r.map(relu, {a.buffer(), b.buffer(), result.buffer()}, p, result.numel());

  return result;
};

Tensor add(Runtime &r, const Tensor &a, const float b, bool with_relu) {

  on_device(r, a);

  Tensor result(r, a.shape(),
                0); // default to zero to avoid random load overhead

  Tensor av = a.broadcast_to(a.shape());
  AddDescScalar p{describe(av), b};

  auto relu = with_relu ? "add_relu_strided_scalar" : "add_strided_scalar";

  r.map(relu, {a.buffer(), result.buffer()}, p, result.numel());

  return result;
};
Tensor pack(Runtime &r, const Tensor &x, bool force_copy) {

  on_device(r, x);

  if (!force_copy && x.contiguous() && x.offset() == 0)
    return x;

  Tensor result(r, x.shape());

  auto p = describe(x);

  r.map("copy_strided", {x.buffer(), result.buffer()}, p, result.numel());
  return result;
}

// relu
Tensor relu(Runtime &r, const Tensor &x) {
  on_device(r, x);

  Tensor result(r, x.shape());

  auto p = describe(x);

  r.map("relu_strided", {x.buffer(), result.buffer()}, p, result.numel());

  return result;
}

// multiply

// scalar multiply
Tensor multiply(Runtime &r, const Tensor &a, const float b, bool with_relu) {

  on_device(r, a);

  Tensor result(r, a.shape(),
                0); // default to zero to avoid random load overhead

  Tensor av = a.broadcast_to(a.shape());
  AddDescScalar p{describe(av), b};

  auto relu = with_relu ? "mul_relu_strided_scalar" : "mul_strided_scalar";

  r.map(relu, {a.buffer(), result.buffer()}, p, result.numel());

  return result;
};
} // namespace minimlx
