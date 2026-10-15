
#include <climits>
#include <random>
#include <vector>
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

// heaaderrs
#include "Metal/Metal.hpp"
#include "story/metal.hh"
#include "story/ops.hh"
#include "story/runtime.hh"
#include "story/tensor.hh"
#include "utility"
#include <cstdint>
#include <memory>
#include <stdexcept>
namespace minimlx {

// helpers
Shape stride_calc(const Shape &shape) {
  uint32_t step = 1;
  Shape strides = std::vector<uint32_t>(shape.size());

  for (int d = shape.size() - 1; d >= 0; --d) {
    strides[d] = step;
    step *= shape[d];
  }

  return strides;
}

// init helper
enum InitType { Random, Value };

#pragma pack(push, 1)
struct InitParams {
  int type;
  size_t n;
  float value;
  short seed;
};
#pragma pack(pop)

void init(Runtime &r, Buffer data, size_t n, InitType type = Random,
          float value = 0) {
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  auto *pipeline = r.pipeline("init");
  InitParams params;
  if (type == 0) {
    std::random_device rd;

    std::mt19937 gen(rd());
    std::uniform_int_distribution<short> dist(-32768, 32767);
    params = {static_cast<int>(type), n, value, dist(gen)};
  } else {
    params = {static_cast<int>(type), n, value, 1};
  }
  LaunchConfig cfg = launch_1d(n);

  r.run("init", {data}, &params, sizeof(params), 1, cfg.threadgroups,
        cfg.threadspergroup);
}

/*
 * CONSTRUTORS
 */
Tensor::Tensor(Runtime &r, std::vector<uint32_t> shape)
    : shape_(std::move(shape)),
      storage_(std::make_shared<Storage>(r, elements(shape_))),
      strides_(std::move(shape_.size())) {
  strides_ = stride_calc(shape_);

  init(r, storage_->buffer, elements(shape_));
};

//
Tensor::Tensor(Runtime &r, Shape shape, float value)
    : shape_(std::move(shape)),
      storage_(std::make_shared<Storage>(r, elements(shape_))),
      strides_(std::move(shape_.size())) {

  InitType type = Value;

  strides_ = stride_calc(shape_);
  init(r, storage_->buffer, elements(shape_), type, value);
};

Tensor Tensor::zeroes(Runtime &r, Shape shape) { return Tensor(r, shape, 0); };

Tensor Tensor::ones(Runtime &r, Shape shape) { return Tensor(r, shape, 1); };

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

// contiguous checks
// checks the shape and strides

bool Tensor::contiguous() const {
  uint32_t expected = 1;

  // checks from the back
  for (int d = int(shape_.size()) - 1; d >= 0; --d) {
    if (shape_[d] != 1 && strides_[d] != expected)
      return false;
    expected *= shape_[d];
  }

  return true;
}

// broadcasting as repeated reading not copies
//  will only work if the broadcasted tensor has either 1 or missing
//  dim in that shape
// achieved by using stride changes to mimic

Tensor Tensor::broadcast_to(Shape target) const {
  elements(target);
  check(target.size() >= shape_.size(),
        "Broadcast target must have at least as many dimensions");

  Tensor view = *this;

  view.shape_ = target;
  view.strides_.assign(target.size(), 0);

  const size_t lead = target.size() - shape_.size();

  for (size_t i = 0; i < shape_.size(); ++i) {
    check(shape_[i] == target[lead + i] || shape_[i] == 1,
          "Broadcasting mismatch");
    view.strides_[lead + i] = shape_[i] == 1 ? 0 : strides_[i];
  }

  view.writable_ = false; // read only as it still points to the
                          // original tensor

  return view;
}

float &Tensor::operator[](size_t i) {
  check(i < numel(), "Index out of bounds");
  auto *p = static_cast<float *>(buffer()->contents());
  return p[i];
}

float Tensor::operator[](size_t i) const {
  check(i < numel(), "Index out of bounds");
  auto *p = static_cast<const float *>(buffer()->contents());
  return p[i];
}

// assign and read

// takes a vector of floats and inputs them
void Tensor::assign(const std::vector<float> &values) {

  check(writable_ && contiguous(), "needs writable contiguous tensors");
  check(values.size() == numel(), "size mismatch");

  auto *p = static_cast<float *>(buffer()->contents()) + offset_;
  std::copy(values.begin(), values.end(), p);
};

std::vector<float> Tensor::read() const {
  std::vector<float> result(numel());

  auto *p = static_cast<const float *>(buffer()->contents());

  for (uint32_t i = 0; i < numel(); ++i) {

    uint32_t q = i, a = offset_;

    for (int d = int(shape_.size()) - 1; d >= 0; --d) {
      a += (q % shape_[d]) * strides_[d];
      q /= shape_[d];
    }
    result[i] = p[a];
  }
  return result;
};

Tensor Tensor::operator+(const Tensor &other) const {
  return add(storage_->runtime, *this, other);
}

} // namespace minimlx
