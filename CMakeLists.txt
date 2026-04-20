cmake_minimum_required(VERSION 3.16)

project(kokkos-extended-precision-demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  string(FIND "${CMAKE_CXX_FLAGS}" "-fext-numeric-literals" _extlit)
  if(_extlit EQUAL -1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals" CACHE STRING "" FORCE)
  endif()
endif()

find_package(Kokkos REQUIRED)

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
  set(QUADMATH_NAMES libquadmath.so quadmath)
  find_library(
    QUADMATH_LIBRARY
    NAMES ${QUADMATH_NAMES}
    PATHS /usr/lib64/atlas
          /usr/lib/atlas
          /usr/lib64
          /usr/lib
          /usr/local/lib64
          /usr/local/lib
          /usr/x86_64-linux-gnu/*
          /usr/lib/gcc/x86_64-linux-gnu/*
          /usr/lib/gcc/x86_64-redhat-linux/*
          /usr/lib/gcc/x86_64-pc-linux-gnu/*
          /soft/compilers/gcc/x86_64-suse-linux/14.1.0/lib64
          "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
  if(QUADMATH_LIBRARY)
    message(STATUS "Found QUADMATH: ${QUADMATH_LIBRARY}")
  else()
    message(FATAL_ERROR "libquadmath not found (required for host __float128 reference)")
  endif()
else()
  message(FATAL_ERROR "This demo currently expects x86_64 + libquadmath for host reference")
endif()

set(THIRD_PARTY_INCLUDE "${CMAKE_SOURCE_DIR}/third_party/include")

add_executable(kokkos_ep_demo src/demo_real.cpp)
target_include_directories(kokkos_ep_demo PRIVATE ${THIRD_PARTY_INCLUDE})
target_link_libraries(kokkos_ep_demo PRIVATE Kokkos::kokkos ${QUADMATH_LIBRARY})

add_executable(kokkos_ep_demo_complex src/demo_complex.cpp)
target_include_directories(kokkos_ep_demo_complex PRIVATE ${THIRD_PARTY_INCLUDE})
target_link_libraries(kokkos_ep_demo_complex PRIVATE Kokkos::kokkos ${QUADMATH_LIBRARY})

install(TARGETS kokkos_ep_demo kokkos_ep_demo_complex RUNTIME DESTINATION bin)
