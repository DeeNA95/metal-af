#include <metal_stdlib>
using namespace metal;

kernel void init(device float *out [[buffer(0)]],
                 constant int8_t *params [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) {

  uint type = ((constant uint *)params)[0];
  uint64_t n = ((constant uint64_t *)(params + 4))[0];
  float value = ((constant float *)(params + 12))[0];
  int seed = ((constant int *)(params + 16))[0];

  if (gid >= n)
    return;

  switch (type) {

  case 0: {

    int x = (gid - seed);
    x = x * 1103515245 + 12345;
    out[gid] = float((x >> 16) & 0x7FFF) / 16384.0f - 1.0f;
    break;
  }
  case 1: {
    out[gid] = value;

    break;
  }
  }
}

kernel void add_serial(device const float *a [[buffer(0)]],
                       device const float *b [[buffer(1)]],
                       device float *out [[buffer(2)]],
                       constant uint &n [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {

  if (gid != 0)
    return;

  for (uint i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}

kernel void add_parallel(device const float *a [[buffer(0)]],
                         device const float *b [[buffer(1)]],
                         device float *out [[buffer(2)]],
                         constant uint &n [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {

  if (gid > n)
    return;

  out[gid] = a[gid] + b[gid];
}

// tensor descriptor
struct TensorDesc {
  uint32_t rank, count, offset, shape[4], strides[4];
};

struct AddDesc {
  TensorDesc a, b;
};
struct AddDescScalar {
  TensorDesc a;
  float b;
};

// given an index and the tensor stride info gets the right value
// uses the general formula for strides
uint tensor_address(uint i, constant TensorDesc &p) {

  uint address = p.offset;

  for (int d = int(p.rank) - 1; d >= 0; d--) {
    address += i % p.shape[d] * p.strides[d];
    i /= p.shape[d];
  }

  return address;
};

kernel void copy_strided(device const float *x [[buffer(0)]],
                         device float *y [[buffer(1)]],
                         constant TensorDesc &p [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {

  if (gid < p.count)
    y[gid] = x[tensor_address(gid, p)];
};

// strided addition kernels
kernel void add_strided(device const float *a [[buffer(0)]],
                        device const float *b [[buffer(1)]],
                        device float *out [[buffer(2)]],
                        constant AddDesc &p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {

  if (gid < p.a.count)

    out[gid] = a[tensor_address(gid, p.a)] + b[tensor_address(gid, p.b)];
}

kernel void add_strided_scalar(device const float *a [[buffer(0)]],
                               device float *out [[buffer(1)]],
                               constant AddDescScalar &p [[buffer(2)]],

                               uint gid [[thread_position_in_grid]]) {

  if (gid < p.a.count)

    out[gid] = a[tensor_address(gid, p.a)] + p.b;
}

kernel void add_relu_strided(device const float *a [[buffer(0)]],
                             device const float *b [[buffer(1)]],
                             device float *out [[buffer(2)]],
                             constant AddDesc &p [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {

  if (gid < p.a.count) {

    out[gid] = a[tensor_address(gid, p.a)] + b[tensor_address(gid, p.b)];

    out[gid] = max(out[gid], 0.0f);
  }
}

kernel void add_relu_strided_scalar(device const float *a [[buffer(0)]],
                                    device float *out [[buffer(1)]],
                                    constant AddDescScalar &p [[buffer(2)]],

                                    uint gid [[thread_position_in_grid]]) {

  if (gid < p.a.count) {

    out[gid] = a[tensor_address(gid, p.a)] + p.b;
    out[gid] = max(out[gid], 0.0f);
  }
}
// relu
kernel void relu_strided(device const float *x [[buffer(0)]],
                         device float *y [[buffer(1)]],
                         constant TensorDesc &p [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {

  if (gid < p.count)

    y[gid] = max(x[tensor_address(gid, p)], 0.0f);
}

// MULTIPLICATION
kernel void mul_strided(device const float *a [[buffer(0)]],
                        device const float *b [[buffer(1)]],
                        device float *out [[buffer(2)]],
                        constant AddDesc &p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {

  if (gid < p.a.count)

    out[gid] = a[tensor_address(gid, p.a)] * b[tensor_address(gid, p.b)];
}

kernel void mul_strided_scalar(device const float *a [[buffer(0)]],
                               device float *out [[buffer(1)]],
                               constant AddDescScalar &p [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
  if (gid < p.a.count)

    out[gid] = a[tensor_address(gid, p.a)] * p.b;
}

kernel void mul_relu_strided(device const float *a [[buffer(0)]],
                             device const float *b [[buffer(1)]],
                             device float *out [[buffer(2)]],
                             constant AddDesc &p [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {

  if (gid < p.a.count) {

    out[gid] =
        max(a[tensor_address(gid, p.a)] * b[tensor_address(gid, p.b)], 0.0f);
  }
}

kernel void mul_relu_strided_scalar(device const float *a [[buffer(0)]],
                                    device float *out [[buffer(1)]],
                                    constant AddDescScalar &p [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]]) {
  if (gid < p.a.count) {

    out[gid] = max(a[tensor_address(gid, p.a)] * p.b, 0.0f);
  }
}
