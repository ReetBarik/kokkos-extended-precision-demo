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

# The host __float128 oracle is provided by Kokkos's quadmath overloads
# (impl/Kokkos_QuadPrecisionMath.hpp), which are compiled and linked when Kokkos
# itself is built with Kokkos_ENABLE_LIBQUADMATH=ON. No separate find_library or
# x86_64 gate is needed here — linking Kokkos::kokkos brings in libquadmath.
# Kokkos exposes its enabled TPLs via the Kokkos_TPLS list (there is no
# Kokkos_ENABLE_LIBQUADMATH variable for TPLs).
if("LIBQUADMATH" IN_LIST Kokkos_TPLS)
  message(STATUS "Kokkos built with LIBQUADMATH: host __float128 oracle enabled")
else()
  message(WARNING
    "Kokkos was NOT built with Kokkos_ENABLE_LIBQUADMATH=ON; the host "
    "__float128 oracle in the demos will not compile. Rebuild Kokkos with "
    "-DKokkos_ENABLE_LIBQUADMATH=ON.")
endif()

# The complex __complex128 oracle in src/demo_complex.cpp routes through a LOCAL
# Kokkos extension header, impl/Kokkos_ComplexQuadPrecisionMath.hpp, which is NOT
# upstream in Kokkos (see patches/README.md). Probe the Kokkos install for it and,
# if absent, warn and continue — same graceful-degradation posture used above for
# LIBQUADMATH itself. Only kokkos_ep_demo_complex depends on this header.
if("LIBQUADMATH" IN_LIST Kokkos_TPLS)
  include(CheckCXXSourceCompiles)
  get_target_property(_kokkos_inc Kokkos::kokkos INTERFACE_INCLUDE_DIRECTORIES)
  set(CMAKE_REQUIRED_INCLUDES ${_kokkos_inc})
  set(CMAKE_REQUIRED_FLAGS "-fext-numeric-literals")
  # Link against Kokkos so libquadmath (the ::c<fn>q symbols the wrapper forwards
  # to) resolves, exactly as kokkos_ep_demo_complex does. Without this the probe
  # compiles but fails to link (undefined reference to cexpq), a false negative.
  set(CMAKE_REQUIRED_LIBRARIES Kokkos::kokkos)
  check_cxx_source_compiles("
    #include <impl/Kokkos_ComplexQuadPrecisionMath.hpp>
    int main() {
      __complex128 z; __real__ z = 1.0Q; __imag__ z = 2.0Q;
      __complex128 w = Kokkos::exp(z);
      return (int)Kokkos::real(w) * 0;
    }" KOKKOS_HAS_COMPLEX_QUADMATH_WRAPPER)
  unset(CMAKE_REQUIRED_INCLUDES)
  unset(CMAKE_REQUIRED_FLAGS)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(KOKKOS_HAS_COMPLEX_QUADMATH_WRAPPER)
    message(STATUS
      "Kokkos install has impl/Kokkos_ComplexQuadPrecisionMath.hpp: "
      "complex __complex128 oracle enabled")
  else()
    message(WARNING
      "Kokkos install is MISSING impl/Kokkos_ComplexQuadPrecisionMath.hpp; "
      "kokkos_ep_demo_complex will not compile its __complex128 oracle. This "
      "header is a local extension carried in this repo — apply it per "
      "patches/README.md, then rebuild Kokkos with -DKokkos_ENABLE_LIBQUADMATH=ON.")
  endif()
endif()

set(THIRD_PARTY_INCLUDE "${CMAKE_SOURCE_DIR}/third_party/include")

add_executable(kokkos_ep_demo src/demo_real.cpp)
target_include_directories(kokkos_ep_demo PRIVATE ${THIRD_PARTY_INCLUDE})
target_link_libraries(kokkos_ep_demo PRIVATE Kokkos::kokkos)

add_executable(kokkos_ep_demo_complex src/demo_complex.cpp)
target_include_directories(kokkos_ep_demo_complex PRIVATE ${THIRD_PARTY_INCLUDE})
target_link_libraries(kokkos_ep_demo_complex PRIVATE Kokkos::kokkos)

install(TARGETS kokkos_ep_demo kokkos_ep_demo_complex RUNTIME DESTINATION bin)

