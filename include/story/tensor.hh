#pragma once

#include "Metal/MTLDevice.hpp"
#include "Metal/MTLResource.hpp"
#include "metal.hh"
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
inline uint32_t elements(const Shape &shape) {
  check(!shape.empty(), "rank 0");
  uint64_t n = 1;
  for (auto s : shape) {
    check(s > 0, "shape must be positive");
    n *= s;
  }
  return n;
};
struct Storage {

  Buffer buffer;
  uint32_t capacity;

  Storage(MTL::Device *device, uint32_t n) : capacity(n) {

    if (!device)
      throw std::runtime_error("DEVICE NOT FOUND OR SHAPE INCORRECT");

    buffer = NS::TransferPtr(
        device->newBuffer(n * sizeof(float), MTL::ResourceStorageModeShared));

    // check if buffer is valid
    check(buffer.get(), "Failed to allocate buffer");
    std::memset(buffer->contents(), 0, n * sizeof(float));
  }
};

class Tensor {
private:
  std::shared_ptr<Storage> storage_;
  Shape shape_, strides_;
  uint32_t offset_ = 0;
  bool writable_ = true;

public:
  // CONSTRUCTORS
  Tensor(MTL::Device *device, Shape shape);

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

  // SHAPE PLAY METHODS
  Tensor transpose(uint32_t a, uint32_t b) const;
  Tensor narrow(uint32_t axis, uint32_t start, uint32_t count) const;
  Tensor reshape(Shape shape) const;
  Tensor broadcast_to(Shape target) const;
};

// struct Tensor {
//
//   // CONSTRUTORS
//   Tensor(MTL::Device *device, uint32_t n);
//
//   Tensor(MTL::Device *device, std::vector<uint32_t> shape);
//
//   // shape
//   const std::vector<uint32_t> &shape() const { return shape_; };
//
//   // indexing
//   float &at(uint32_t row, uint32_t col);
//
//   float &at(uint32_t i);
//
//   float &operator[](uint32_t i);
//
//   const float &operator[](uint32_t i) const;
//
//   // number of elements
//   uint32_t numel() const;
//
//   // get data on cpu
//   float *data();
//
//   const float *data() const;
//
//   MTL::Buffer *buffer() const;
//   inline uint32_t elements(const Shape &shape) {
//     check(!shape.empty(), "rank 0");
//     uint64_t n = 1;
//     for (auto s : shape) {
//       check(s > 0, "shape must be positive");
//       n *= s;
//     }
//     return n;
//   };
//
// private:
//   Buffer storage_;
//
//   uint32_t count_;
//   std::vector<uint32_t> shape_;
// };
}; // namespace minimlx
