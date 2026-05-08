#!/usr/bin/env bash
################################################################
# Usage: source scripts/build_with_kokkos.sh <install-dir>     #
#      Ensure your environment has compilers and CUDA/HIP/etc  #
#      drivers for your target architecture loaded first.      #
################################################################
# Builds Kokkos into <install-dir>, then configures and builds #
# this repository under <repo>/build/.                         #
#                                                              #
# To target a different backend, compiler, or GPU, uncomment   #
# the appropriate block in the SETTINGS section below.         #
################################################################

# install in provided path or in current path
export TARGET_DIR=$1
if [ "$#" -ne 1 ]; then
  export TARGET_DIR=$(pwd -LP)
fi
START_DIR=$(pwd -LP)

# Resolve repo root while cwd is still the caller's directory. After `cd "$TARGET_DIR"`
# below, a relative ${BASH_SOURCE[0]} would break (e.g. dirname becomes build/scripts).
_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=$(cd "${_SCRIPT_DIR}/.." && pwd)

echo "Installing Kokkos into: $TARGET_DIR"

mkdir -p "$TARGET_DIR"

cd "$TARGET_DIR" || exit 1

export LOGDIR=$TARGET_DIR/build_logs
mkdir -p "$LOGDIR"

##############
## SETTINGS ##
##############

# Compiler settings (choose one):
# A) generic GCC
CC=$(which gcc)
CXX=$(which g++)
# B) possible alternative on AMD systems:
# CC=$(which hipcc)
# CXX=$(which hipcc)
# C) possible alternative on Cray systems:
# CC=$(which cc)
# CXX=$(which CC)
# D) possible alternative on Intel systems:
# CC=$(which icx)
# CXX=$(which icpx)


# KOKKOS Related settings
KOKKOS_TAG=5.1.0
KOKKOS_BUILD=Release
KOKKOS_URL=https://github.com/kokkos/kokkos.git

# Enable Software Framework (choose one):
# A) Enable Serial (CPU only)
KOKKOS_ENABLED=Kokkos_ENABLE_SERIAL
# B) Enable CUDA
# KOKKOS_ENABLED=Kokkos_ENABLE_CUDA
# C) Enable HIP
# KOKKOS_ENABLED=Kokkos_ENABLE_HIP
# D) Enable OpenMP
# KOKKOS_ENABLED=Kokkos_ENABLE_OPENMP
# E) Enable SYCL
# KOKKOS_ENABLED=Kokkos_ENABLE_SYCL
# more available on Kokkos website

# Enable Architecture (choose one):
# A) Serial CPU (no specific architecture needed)
KOKKOS_ARCH_FLAG=NONE
# B) NVidia GB200 / B100 (sm_100, default for this repo)
# KOKKOS_ARCH_FLAG=Kokkos_ARCH_BLACKWELL100
# C) NVidia H100
# KOKKOS_ARCH_FLAG=Kokkos_ARCH_HOPPER90
# D) NVidia A100
# KOKKOS_ARCH_FLAG=Kokkos_ARCH_AMPERE80
# E) NVidia V100
# KOKKOS_ARCH_FLAG=Kokkos_ARCH_VOLTA70
# F) AMD MI250
# KOKKOS_ARCH_FLAG=Kokkos_ARCH_VEGA90A
# G) AMD MI100
# KOKKOS_ARCH_FLAG=Kokkos_ARCH_VEGA908
# H) Intel Skylake CPU
# KOKKOS_ARCH_FLAG=Kokkos_ARCH_SKX
# I) Mac (Apple Silicon ARM)
# KOKKOS_ARCH_FLAG=Kokkos_ARCH_ARMV80
# J) Intel PVC
# KOKKOS_ARCH_FLAG=Kokkos_ARCH_INTEL_PVC
# more available on Kokkos website

# CUDA architecture number to pass to CMAKE_CUDA_ARCHITECTURES.
# Only consulted when KOKKOS_ENABLED=Kokkos_ENABLE_CUDA.
# Match this to the KOKKOS_ARCH_FLAG you selected above:
#   Blackwell sm_100 -> 100   Hopper sm_90 -> 90
#   Ampere   sm_80  -> 80    Volta sm_70 -> 70
CUDA_ARCH_NUMBER=100

# Backend-specific extra flags
NO_EXTRA_FLAGS=
CUDA_EXTRA_FLAGS="-DKokkos_ENABLE_CUDA_LAMBDA=On \
                  -DKokkos_ENABLE_CUDA_CONSTEXPR=On \
                  -DKokkos_ENABLE_CUDA_FASTMATH=Off"
HIP_EXTRA_FLAGS="-DCMAKE_CXX_COMPILER=$CXX \
                 -DCMAKE_CXX_FLAGS=\"--gcc-toolchain=/soft/compilers/gcc/13.3.0/x86_64-suse-linux\""
# Set this to match the backend you chose above:
EXTRA_FLAGS=$NO_EXTRA_FLAGS
# EXTRA_FLAGS=$CUDA_EXTRA_FLAGS
# EXTRA_FLAGS=$HIP_EXTRA_FLAGS


####################
## install Kokkos ##
####################
echo "Installing Kokkos ARCH=$KOKKOS_ARCH_FLAG  BACKEND=$KOKKOS_ENABLED"
{
  git clone "$KOKKOS_URL" -b "$KOKKOS_TAG"
  cd kokkos

  # Pass CMAKE_CUDA_ARCHITECTURES only when CUDA is the active backend.
  CUDA_ARCH_ARG=
  if [ "$KOKKOS_ENABLED" = "Kokkos_ENABLE_CUDA" ]; then
    CUDA_ARCH_ARG="-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH_NUMBER"
  fi

  if [ "$KOKKOS_ARCH_FLAG" = "NONE" ]; then
    cmake -S . -B "build/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD" \
      -DCMAKE_INSTALL_PREFIX="install/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD" \
      -DCMAKE_BUILD_TYPE=$KOKKOS_BUILD \
      -DCMAKE_CXX_STANDARD=20 \
      -D$KOKKOS_ENABLED=ON \
      $EXTRA_FLAGS
  else
    cmake -S . -B "build/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD" \
      -DCMAKE_INSTALL_PREFIX="install/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD" \
      -DCMAKE_BUILD_TYPE=$KOKKOS_BUILD \
      -DCMAKE_CXX_STANDARD=20 \
      -D$KOKKOS_ARCH_FLAG=ON \
      -D$KOKKOS_ENABLED=ON \
      $CUDA_ARCH_ARG \
      $EXTRA_FLAGS
  fi

  make -C "build/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD" -j"$(nproc)" install

  echo "export KOKKOS_HOME=$PWD/install/kokkos-$KOKKOS_TAG/$KOKKOS_BUILD" >setup.sh
  echo "export CMAKE_PREFIX_PATH=\$KOKKOS_HOME/lib64/cmake/Kokkos:\$CMAKE_PREFIX_PATH" >>setup.sh
  echo "export CPATH=\$KOKKOS_HOME/include:\$CPATH" >>setup.sh
  echo "export PATH=\$KOKKOS_HOME/bin:\$PATH" >>setup.sh
  echo "export LD_LIBRARY_PATH=\$KOKKOS_HOME/lib64:\$LD_LIBRARY_PATH" >>setup.sh
  # shellcheck disable=SC1091
  source setup.sh
} 1>"$LOGDIR/kokkos.stdout.txt" 2>"$LOGDIR/kokkos.stderr.txt"

# Fail loudly if the Kokkos install step did not produce setup.sh — otherwise
# the redirect above silently swallows the failure and the project cmake later
# fails with a confusing "find_package(Kokkos)" error.
if [ ! -f "$TARGET_DIR/kokkos/setup.sh" ]; then
  echo "ERROR: Kokkos install failed. See:" >&2
  echo "  $LOGDIR/kokkos.stderr.txt" >&2
  echo "  $LOGDIR/kokkos.stdout.txt" >&2
  return 1 2>/dev/null || exit 1
fi

cd "$TARGET_DIR" || exit 1

ulimit -s 131072

# shellcheck disable=SC1091
source "$TARGET_DIR/kokkos/setup.sh"

mkdir -p "$REPO_ROOT/build"
cd "$REPO_ROOT/build" || exit 1

cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CXX_FLAGS="-g" \
  "$REPO_ROOT"

make -j"$(nproc)"

echo "Executables:"
echo "  $REPO_ROOT/build/kokkos_ff_demo"
echo "  $REPO_ROOT/build/kokkos_ff_demo_complex"
cd ..
