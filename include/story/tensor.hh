#pragma once
#include "metal.hh"
#include "runtime.hh"
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace minimlx {

// helpers for stride
using Shape = std::vector<uint32_t>;

inline void check(bool ok, const char *text) {
  if (!ok)
    throw std::invalid_argument(text);
};
inline uint64_t elements(const Shape &shape) {
  check(!shape.empty(), "rnnk 0");
  uint64_t n = 1;
  for (auto s : shape) {
    check(s > 0, "shape must be positive");
    n *= s;
  }
  return n;
};
struct Storage {
  Runtime &runtime;
  Buffer buffer;
  uint32_t capacity;

  Storage(Runtime &r, uint64_t n) : capacity(n), runtime(r) {

    if (!r.device.get())
      throw std::runtime_error("DEVICE NOT FOUND OR SHAPE INCORRECT");

    buffer = NS::TransferPtr(
        r.device->newBuffer(n * sizeof(float), MTL::ResourceStorageModeShared));

    // check if buffer is valid
    check(buffer.get(), "Failed to allocate buffer");
  }
};

/*
 * TENSOR CLASS
 */

class Tensor {
private:
  Shape shape_, strides_;
  uint32_t offset_ = 0;
  bool writable_ = true;

  std::shared_ptr<Storage> storage_;

public:
  // CONSTRUCTORS
  Tensor(Runtime &r, Shape shape);
  Tensor(Runtime &r, Shape shape, float value);
  static Tensor zeroes(Runtime &r, Shape shape);
  static Tensor ones(Runtime &r, Shape shape);

  const Shape &shape() const;
  const Shape &strides() const;

  uint32_t offset() const;
  uint32_t numel() const;

  const Buffer &buffer() const;

  bool writable() const;

  uint32_t address(const Shape &index) const;

  float at(const Shape &index) const;

  void set(const Shape &index, float value);

  bool contiguous() const;

  void assign(const std::vector<float> &values);

  std::vector<float> read() const;

  float &operator[](size_t i);
  float operator[](size_t i) const;
  Tensor operator+(const Tensor &other) const;

  // SHAPE PLAY METHODS
  Tensor transpose(uint32_t a, uint32_t b) const;
  Tensor narrow(uint32_t axis, uint32_t start, uint32_t count) const;
  Tensor reshape(Shape shape) const;
  Tensor broadcast_to(Shape target) const;
};

// tensor descriptor
struct TensorDesc {
  uint32_t rank, count, offset, shape[4], strides[4];
};

static_assert(sizeof(TensorDesc) == 44);

inline TensorDesc describe(const Tensor &t) {
  TensorDesc p{};

  p.rank = uint32_t(t.shape().size());
  p.count = t.numel();
  p.offset = t.offset();

  for (uint32_t d = 0; d < p.rank; ++d) {
    p.shape[d] = t.shape()[d];
    p.strides[d] = t.strides()[d];
  }

  return p;
};

struct AddDesc {
  TensorDesc a, b;
};
struct AddDescScalar {
  TensorDesc a;
  float b;
};

static_assert(sizeof(AddDesc) == 88);
}; // namespace minimlx
   //
