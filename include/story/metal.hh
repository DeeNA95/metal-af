#pragma once
#include <Metal/Metal.hpp>

#include <Foundation/Foundation.hpp>
#include <Foundation/NSSharedPtr.hpp>

#include <cstdint>
#include <algorithm>
using Buffer = NS::SharedPtr<MTL::Buffer>;

constexpr uint32_t DEFAULT_GROUP_SIZE = 256;

struct LaunchConfig {
  MTL::Size threadgroups;
  MTL::Size threadspergroup;
};

inline LaunchConfig launch_1d(uint32_t n) {
  uint32_t groups = (n + DEFAULT_GROUP_SIZE - 1) / DEFAULT_GROUP_SIZE;
  return {
    MTL::Size(groups, 1, 1),
    MTL::Size(DEFAULT_GROUP_SIZE, 1, 1)
  };
}

inline LaunchConfig launch_2d(uint32_t w, uint32_t h) {
  uint32_t group_w = std::min(w, DEFAULT_GROUP_SIZE);
  uint32_t group_h = std::min(h, DEFAULT_GROUP_SIZE);
  uint32_t gw = (w + group_w - 1) / group_w;
  uint32_t gh = (h + group_h - 1) / group_h;
  return {
    MTL::Size(gw, gh, 1),
    MTL::Size(group_w, group_h, 1)
  };
}
