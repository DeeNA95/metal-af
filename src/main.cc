#include "story/metal.hh"
#include "story/ops.hh"
#include "story/runtime.hh"
#include "story/tensor.hh"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace minimlx;

void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: main_check <library> [kernel]\n";
    return 1;
  }
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  Runtime runtime("build/kernels/tensor.metallib");

  const char *ker_name = argc > 2 ? argv[2] : "add_parallel";

  printf("Using device: %s\n", runtime.device->name()->utf8String());

  uint32_t n = 4096 * 2048;
  if (n % 1024 != 0) {
    throw std::runtime_error("n must be a multiple of 1024");
  };

  std::vector<uint32_t> shape = {n / 1024, 1024};
  printf("\nTensors have shape %d,%d\n\n", shape[0], shape[1]);

  Tensor x(runtime, shape), y(runtime, shape);

  auto t0 = std::chrono::high_resolution_clock::now();
  Tensor output = add_cpu(runtime, x, y);
  auto t1 = std::chrono::high_resolution_clock::now();
  auto cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "CPU add: " << cpu_ms << " ms\n";

  // GPU serial
  //
  Tensor output_gpu(runtime, x.shape());

  printf("Before transpose x[2,3] = %.3f\n", x.at({2, 3}));
  auto transposed = x.transpose(0, 1);

  printf("After transpose x[2,3] = %.3f\n", transposed.at({2, 3}));

  Tensor zeros = Tensor::zeroes(runtime, shape);
  Tensor one = Tensor::ones(runtime, shape);
  printf("zeros[2,3] = %.3f\n", zeros.at({2, 3}));
  printf("ones[2,3] = %.3f\n", one.at({2, 3}));

  struct AddParams {
    uint32_t n;
  };
  AddParams a = {n};
  LaunchConfig cfg = launch_1d(n);

  auto time =
      runtime.run("add_parallel", {x.buffer(), y.buffer(), output_gpu.buffer()},
                  &a, sizeof(a), 3, cfg.threadgroups, cfg.threadspergroup);

  printf("GPU add took %.3fms\n", time);
  printf("Given x = %.3f and y = %.3f: GPU add: %.3f\n", x.at({2, 3}),
         y.at({2, 3}), output_gpu.at({2, 3}));

  printf("TEST WITH + \n");

  Tensor ouput_gpu2 = x + y;

  printf("Given x = %.3f and y = %.3f: GPU add: %.3f\n", x.at({2, 3}),
         y.at({2, 3}), ouput_gpu2.at({2, 3}));

  printf("testing with needed broad casting\n");
  Tensor x2(runtime, {1, 2048, 512}), y2(runtime, {57, 2048, 1});
  Tensor output_gpu3 = x2 + y2;
  printf("Given x= %.3f and y = %.3f: GPU add: %.3f\n", x2.at({0, 3, 0}),

         y2.at({0, 3, 0}), output_gpu3.at({0, 3, 0}));

  printf("testing relu add\n");

  Tensor output_gpu4 = add(runtime, x, y, true);
  printf("Given x= %.3f and y = %.3f: GPU add: %.3f\n", x.at({2, 3}),
         y.at({2, 3}), output_gpu4.at({2, 3}));

  printf("testing relu add scalar \n");
  Tensor output_gpu5 = add(runtime, x, 0.5, true);
  printf("Given x= %.3f and y = %f : GPU add: %.3f\n", x.at({2, 3}), 0.5,
         output_gpu5.at({2, 3}));

  printf("testing relu mul scalar \n");
  Tensor output_gpu6 = multiply(runtime, x, 0.5, true);
  printf("Given x= %.3f and y = %f : GPU add: %.3f\n", x.at({2, 3}), 0.5,
         output_gpu6.at({2, 3}));

  return 0;
}
