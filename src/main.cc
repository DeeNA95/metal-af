#include "Foundation/NSError.hpp"
#include "Foundation/NSSharedPtr.hpp"
#include "Foundation/NSString.hpp"
#include "story/metal.hh"
#include "story/ops.hh"
#include "story/tensor.hh"
#include <chrono>
#include <cstdint>
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

  // todo
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  require(device.get(), "Metal device not found");

  // Create a command queue
  auto queue = NS::TransferPtr(device->newCommandQueue());
  require(queue.get(), "Cannot create command queue\n");

  // create an error so if the lib load fails it point sothe err
  NS::Error *error = nullptr;

  const char *path = argv[1];
  auto lib = NS::TransferPtr(device->newLibrary(
      NS::String::string(path, NS::UTF8StringEncoding), &error

      ));
  require(lib.get(), error ? error->localizedDescription()->utf8String()
                           : "Cannot load lib");

  const char *ker_name = argc > 2 ? argv[2] : "add_parallel";

  auto func = NS::TransferPtr(
      lib->newFunction(NS::String::string(ker_name, NS::UTF8StringEncoding)));
  require(func.get(), "Cannot create function");

  auto pipeline =
      NS::TransferPtr(device->newComputePipelineState(func.get(), &error));
  require(pipeline.get(), "cannot create pipeline");

  if (!device.get())
    throw std::runtime_error("No Metal device");

  printf("Using device: %s\n", device->name()->utf8String());

  uint32_t n = 8293376;
  if (n % 1024 != 0) {
    throw std::runtime_error("n must be a multiple of 1024");
  };

  std::vector<uint32_t> shape = {n / 1024, 1024};
  printf("\nTensors have shape %d,%d\n\n", shape[0], shape[1]);

  Tensor x(device.get(), shape), y(device.get(), shape);

  // Create a random number generator
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Fill x and y with random values
  for (int i = 0; i < x.numel(); i++) {
    x[i] = dist(rng);
    y[i] = dist(rng);
  }

  float *p = x.data();
  p[2] = 9;
  std::cout << x.at(9, 835) << '\n';

  auto t0 = std::chrono::high_resolution_clock::now();
  Tensor output = add_cpu(device.get(), x, y);
  auto t1 = std::chrono::high_resolution_clock::now();
  auto cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "CPU add: " << cpu_ms << " ms\n";

  // GPU serial
  auto *command = queue->commandBuffer();
  auto *encoder = command->computeCommandEncoder();

  Tensor output_gpu(device.get(), x.numel());

  encoder->setComputePipelineState(pipeline.get());
  encoder->setBuffer(x.buffer(), 0, 0);
  encoder->setBuffer(y.buffer(), 0, 1);
  encoder->setBuffer(output_gpu.buffer(), 0, 2);
  encoder->setBytes(&n, sizeof(n), 3);

  const auto threads = std::string_view(ker_name) == "add_parallel"
                           ? pipeline->threadExecutionWidth()
                           : 1;

  const auto groups = std::string_view(ker_name) == "add_parallel"
                          ? (n + threads - 1) / threads
                          : 1;

  encoder->dispatchThreadgroups(MTL::Size(groups, 1, 1),
                                MTL::Size(threads, 1, 1));
  encoder->endEncoding();

  t0 = std::chrono::high_resolution_clock::now();
  command->commit();
  command->waitUntilCompleted();
  t1 = std::chrono::high_resolution_clock::now();

  auto serial_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "GPU add : " << serial_ms << " ms\n";
  require(command->status() != MTL::CommandBufferStatusError,
          "command buffer failed");
}
