#include "story/metal.hh"
#include <memory>
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Metal/MTLDevice.hpp"
#include "story/tensor.hh"
#include "utility"
#include <cstdint>
#include <stdexcept>

namespace minimlx {

Shape stride_calc(const Shape &shape) {
  uint32_t step = 1;
  Shape strides = std::vector<uint32_t>(shape.size());

  for (int d = shape.size() - 1; d >= 0; --d) {
    strides[d] = step;
    step *= shape[d];
  }

  return strides;
}
// CONSTRUTORS
Tensor::Tensor(MTL::Device *device, std::vector<uint32_t> shape)
    : shape_(std::move(shape)),
      storage_(std::make_shared<Storage>(device, elements(shape))),
      strides_(std::move(shape_.size())) {
  strides_ = stride_calc(shape);
};

const Shape &Tensor::shape() const { return shape_; };

const Shape &Tensor::strides() const { return strides_; };

uint32_t Tensor::offset() const { return offset_; };

uint32_t Tensor::numel() const { return elements(shape_); };

const Buffer &Tensor::buffer() const { return storage_->buffer; };

bool Tensor::writable() const { return writable_; };

// read and write

uint32_t Tensor::address(const Shape &index) const {
  check(index.size() == shape_.size(), "Coordinate rank mismatch");

  uint64_t a = offset_;

  for (int d = 0; d < index.size(); ++d) {
    check(index[d] < shape_[d], "Coordinate outside tensor");
    a += uint64_t(index[d] * strides_[d]);
  }

  check(a < storage_->capacity, "Address outside storage");
  return uint32_t(a);
}

float Tensor::at(const Shape &index) const {
  auto *p = static_cast<const float *>(buffer()->contents());

  return p[address(index)];
}

void Tensor::set(const Shape &index, float value) {
  check(writable_, "Tensor is not writable");
  auto *p = static_cast<float *>(buffer()->contents());

  p[address(index)] = value;
}

// SHAPE PLAY METHODS

Tensor Tensor::reshape(Shape new_shape) const {
  check(contiguous(), "Contiguous tensor required for reshape");
  check(elements(new_shape) == numel(), "Number of elements mismatch");

  Tensor view = *this; // like python self

  // get the new shape and strides
  view.shape_ = std::move(new_shape);
  view.strides_.resize(view.shape_.size());

  view.strides_ = stride_calc(view.shape_);

  return view;
}

Tensor Tensor::transpose(uint32_t a, uint32_t b) const {
  check(a < shape_.size() && b < shape_.size(), "Invalid transpose axes");
  Shape new_shape = shape_;
  std::swap(new_shape[a], new_shape[b]);
  return reshape(std::move(new_shape));
}

Tensor Tensor::narrow(uint32_t axis, uint32_t start, uint32_t count) const {
  check(start + count <= shape_[axis], "Narrow out of bounds");
  check(axis < shape_.size() && count > 0, "Invalid slice");

  Tensor view = *this;
  view.offset_ += start * strides_[axis];
  view.shape_[axis] = count;
  return view;
}

} // namespace minimlx
