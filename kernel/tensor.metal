#include <metal_stdlib>
using namespace metal;

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
