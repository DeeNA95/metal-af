#pragma once

#include "metal.hh"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct Runtime {
  NS::SharedPtr<MTL::Device> device;
  NS::SharedPtr<MTL::CommandQueue> queue;
  NS::SharedPtr<MTL::Library> library;
  std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>>
      cache;
  NS::SharedPtr<MTL::CommandBuffer> active_batch;

  explicit Runtime(const char *path) {
    device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!device.get())
      throw std::runtime_error("No Metal GPU device");

    queue = NS::TransferPtr(device->newCommandQueue());
    if (!queue.get())
      throw std::runtime_error("Cannot create queue");

    NS::Error *error = nullptr;
    library = NS::TransferPtr(device->newLibrary(
        NS::String::string(path, NS::UTF8StringEncoding), &error));
    if (!library.get())
      throw std::runtime_error(error
                                   ? error->localizedDescription()->utf8String()
                                   : "Cannot load metallib");
  }

  MTL::ComputePipelineState *pipeline(const std::string &name) {
    if (!cache.count(name)) {
      auto f = NS::TransferPtr(library->newFunction(
          NS::String::string(name.c_str(), NS::UTF8StringEncoding)));
      if (!f.get())
        throw std::runtime_error("Unknown kernel: " + name);

      NS::Error *error = nullptr;
      auto p =
          NS::TransferPtr(device->newComputePipelineState(f.get(), &error));
      if (!p.get())
        throw std::runtime_error(
            error ? error->localizedDescription()->utf8String()
                  : "Pipeline creation failed");

      cache.emplace(name, std::move(p));
    }

    return cache.at(name).get();
  }

  template <class T> Buffer buffer(const std::vector<T> &values) {
    auto b = NS::TransferPtr(device->newBuffer(
        std::max<size_t>(sizeof(T), values.size() * sizeof(T)),
        MTL::ResourceStorageModeShared));
    if (!b.get())
      throw std::runtime_error("Buffer allocation failed");

    if (!values.empty())
      std::memcpy(b->contents(), values.data(), values.size() * sizeof(T));

    return b;
  }

  Buffer zeros(size_t n) { return buffer(std::vector<float>(n, 0)); }

  template <class T> std::vector<T> read(const Buffer &b, size_t n) {
    // The caller must synchronize pending GPU work before CPU access.
    const T *p = static_cast<const T *>(b->contents());
    return std::vector<T>(p, p + n);
  }

  void begin_batch() {
    synchronize();
    active_batch = NS::RetainPtr(queue->commandBuffer());
  }

  double synchronize() {
    if (!active_batch.get())
      return 0;

    auto batch = std::move(active_batch);
    batch->commit();
    batch->waitUntilCompleted();
    if (batch->status() == MTL::CommandBufferStatusError)
      throw std::runtime_error(
          batch->error()->localizedDescription()->utf8String());

    return 1e6 * (batch->GPUEndTime() - batch->GPUStartTime());
  }

  double run(const std::string &name, const std::vector<Buffer> &buffers,
             const void *params, size_t param_bytes, uint32_t param_index,
             MTL::Size groups, MTL::Size threads) {
    auto *p = pipeline(name);

    size_t count = threads.width * threads.height * threads.depth;
    if (!count || count > p->maxTotalThreadsPerThreadgroup())
      throw std::runtime_error("Illegal threadgroup size");

    auto *command =
        active_batch.get() ? active_batch.get() : queue->commandBuffer();
    auto *encoder = command->computeCommandEncoder();

    encoder->setComputePipelineState(p);

    for (size_t i = 0; i < buffers.size(); ++i)
      encoder->setBuffer(buffers[i].get(), 0, i);

    if (param_bytes)
      encoder->setBytes(params, param_bytes, param_index);

    encoder->dispatchThreadgroups(groups, threads);
    encoder->endEncoding();

    if (active_batch.get())
      return 0;

    command->commit();
    command->waitUntilCompleted();

    if (command->status() == MTL::CommandBufferStatusError)
      throw std::runtime_error(
          command->error()->localizedDescription()->utf8String());
    // report time in ms
    return 1e3 * (command->GPUEndTime() - command->GPUStartTime());
  }

  template <class P>
  double map(const std::string &name, const std::vector<Buffer> &b, const P &p,
             uint32_t n) {
    if (!n)
      return 0; // dispatching zero work is handled on the host

    auto cfg = launch_1d(n);

    return run(name, b, &p, sizeof(p), uint32_t(b.size()), cfg.threadgroups,

               cfg.threadspergroup);
  }

  template <class P>
  double rows(const std::string &name, const std::vector<Buffer> &b, const P &p,
              uint32_t row_count) {
    auto width = pipeline(name)->threadExecutionWidth();
    return run(name, b, &p, sizeof(p), uint32_t(b.size()),
               MTL::Size(row_count, 1, 1), MTL::Size(width, 1, 1));
  }
};
