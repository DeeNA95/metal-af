cmake_minimum_required(VERSION 4.4.3)
project(METALSTUDY CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

# METAL INTERFACE

add_library(metal_cpp INTERFACE)
target_include_directories(
  metal_cpp
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/metal-cpp>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/metal-cpp>")

target_link_libraries(metal_cpp INTERFACE "-framework Metal"
                                          "-framework Foundation")

# runtime lib
add_library(
  minimlx
  # add .cc files as they appear
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ops.cc)

target_link_libraries(minimlx PUBLIC metal_cpp)

# alias all kernel names to kernel sources to compile as one
set(KERNELS kernel/tensor.metal)

# takes each kernel in the kernels set and compiles to air then metallib
foreach(kernel ${KERNELS})
  get_filename_component(base ${kernel} NAME_WE)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernels/${base}.metallib
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_BINARY_DIR}/kernels/
    COMMAND
      xcrun -sdk macosx metal -c -std=metal4.1 -fno-fast-math
      ${CMAKE_CURRENT_SOURCE_DIR}/${kernel} -o
      ${CMAKE_CURRENT_BINARY_DIR}/kernels/${base}.air
    COMMAND
      xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/kernels/${base}.air
      -o ${CMAKE_CURRENT_BINARY_DIR}/kernels/${base}.metallib
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${kernel}
    COMMENT "COMPILING ${kernel}")
  add_custom_target(
    ${base}_metallib ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kernels/${base}.metallib)
endforeach()

# tests will go here general make check
add_executable(main_check src/main.cc)
target_link_libraries(main_check PRIVATE minimlx)

# install rules
install(
  TARGETS minimlx metal_cpp
  EXPORT minimlxTargets
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/story DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  DIRECTORY third_party/metal-cpp/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/metal-cpp
  PATTERN ".git" EXCLUDE)
install(
  EXPORT minimlxTargets
  FILE minimlxTargets.cmake
  NAMESPACE minimlx::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/minimlx)
