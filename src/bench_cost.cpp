// bench_cost — emulation cost of each backend against its precision-tier peer.
//
// The question this answers is NOT "how much slower than FP64 is extended
// precision" — FP64 is not a substitute for a 29- or 31-digit type, so that
// ratio compares things nobody chooses between. Each backend is instead timed
// against the incumbent that delivers comparable precision:
//
//   FF (~14 digits)  ->  FP64 (~16 digits)
//        FF reconstructs FP64-class precision out of FP32, for hardware where
//        FP64 is absent or rate-limited. FP64 is the thing it substitutes for.
//
//   DD (~31 digits)  ->  __float128 / libquadmath (~34 digits)
//   QF (~29 digits)  ->  __float128 / libquadmath (~34 digits)
//        These are extended-precision types. libquadmath is the incumbent
//        software path at that precision, so it is the honest comparator.
//
// Inputs are converted to each backend's type BEFORE timing, so what is
// measured is the operation, not the cost of constructing the type.
//
// Caveat worth carrying into any comparison: __float128 is a full IEEE
// binary128 implementation with correct rounding, subnormals, and exception
// semantics. DD and QF are unevaluated multi-word expansions with a narrower
// exponent range and no correct-rounding guarantee. A speed win against
// libquadmath is real but is not free of trade-offs.
//
// SPDX-License-Identifier: Apache-2.0

#include <quadmath.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "dd_math.hpp"
#include "ff_math.hpp"
#include "qf_math.hpp"

namespace ex = Kokkos::Experimental;
using clock_type = std::chrono::steady_clock;

namespace {

constexpr int kDefaultBatch   = 20000;
constexpr int kDefaultRepeats = 5;
constexpr unsigned long long kDefaultSeed = 12345ULL;
constexpr int kWarmupRuns = 1;

volatile double g_sink = 0.0;

struct Config {
  int batch = kDefaultBatch;
  int repeats = kDefaultRepeats;
  unsigned long long seed = kDefaultSeed;
  std::string op;  // empty => all
};

// Median wall time in ms over `repeats` runs, after a warmup.
template <class F>
double time_median_ms(int repeats, F&& body) {
  for (int w = 0; w < kWarmupRuns; ++w) body();
  std::vector<double> t;
  t.reserve(static_cast<size_t>(repeats));
  for (int r = 0; r < repeats; ++r) {
    auto t0 = clock_type::now();
    body();
    t.push_back(std::chrono::duration<double, std::milli>(clock_type::now() - t0).count());
  }
  std::sort(t.begin(), t.end());
  const size_t n = t.size();
  return (n % 2 == 1) ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
}

struct Row {
  const char* op;
  const char* cls;  // "arith" | "trans"
  double f64, q128, dd, ff, qf;
};

}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "error: %s needs a value\n", what); std::exit(2); }
      return argv[++i];
    };
    if      (a == "--batch")   cfg.batch   = std::atoi(next("--batch"));
    else if (a == "--repeats") cfg.repeats = std::atoi(next("--repeats"));
    else if (a == "--seed")    cfg.seed    = std::strtoull(next("--seed"), nullptr, 10);
    else if (a == "--op")      cfg.op      = next("--op");
    else if (a == "--help") {
      std::printf(
          "Usage: %s [--op <name>] [--batch N] [--repeats N] [--seed N]\n"
          "  Ops: add sub mul div sqrt fma | exp log sin cos atan pow\n"
          "  Defaults: batch=%d repeats=%d seed=%llu\n",
          argv[0], kDefaultBatch, kDefaultRepeats, kDefaultSeed);
      return 0;
    } else { std::fprintf(stderr, "error: unknown argument %s\n", a.c_str()); return 2; }
  }
  if (cfg.batch <= 0 || cfg.repeats <= 0) {
    std::fprintf(stderr, "error: --batch and --repeats must be positive\n");
    return 2;
  }

  Kokkos::initialize(argc, argv);
  {
    const int n = cfg.batch;
    std::mt19937_64 gen(cfg.seed);
    std::uniform_real_distribution<double> dist(0.5, 100.0);

    std::vector<double> xa(n), xb(n);
    for (int i = 0; i < n; ++i) { xa[i] = dist(gen); xb[i] = dist(gen); }

    // Pre-convert: the timed region measures the op, not the conversion.
    std::vector<__float128>      qa(n), qb(n);
    std::vector<ex::DoubleDouble> da(n), db(n);
    std::vector<ex::FloatFloat>   fa(n), fb(n);
    std::vector<ex::QuadFloat>    ka(n), kb(n);
    for (int i = 0; i < n; ++i) {
      qa[i] = (__float128)xa[i];      qb[i] = (__float128)xb[i];
      da[i] = ex::DoubleDouble(xa[i]); db[i] = ex::DoubleDouble(xb[i]);
      fa[i] = ex::FloatFloat(xa[i]);   fb[i] = ex::FloatFloat(xb[i]);
      ka[i] = ex::QuadFloat(xa[i]);    kb[i] = ex::QuadFloat(xb[i]);
    }

    // Each op is five loops over the same data, one per type. Written out
    // rather than templated so every backend's expression is visible and no
    // implicit conversion sneaks into a timed region.
#define BENCH(NAME, CLS, F64EXPR, QEXPR, DDEXPR, FFEXPR, QFEXPR)                 \
  do {                                                                           \
    if (!cfg.op.empty() && cfg.op != NAME) break;                                \
    Row r{NAME, CLS, 0, 0, 0, 0, 0};                                             \
    r.f64 = time_median_ms(cfg.repeats, [&] {                                    \
      double s = 0; for (int i = 0; i < n; ++i) { s += (F64EXPR); } g_sink = s; });\
    r.q128 = time_median_ms(cfg.repeats, [&] {                                   \
      __float128 s = 0; for (int i = 0; i < n; ++i) { s += (QEXPR); }            \
      g_sink = (double)s; });                                                    \
    r.dd = time_median_ms(cfg.repeats, [&] {                                     \
      ex::DoubleDouble s(0.0); for (int i = 0; i < n; ++i) { s = s + (DDEXPR); } \
      g_sink = s.hi; });                                                         \
    r.ff = time_median_ms(cfg.repeats, [&] {                                     \
      ex::FloatFloat s(0.0f); for (int i = 0; i < n; ++i) { s = s + (FFEXPR); }  \
      g_sink = (double)s.hi; });                                                 \
    r.qf = time_median_ms(cfg.repeats, [&] {                                     \
      ex::QuadFloat s(0.0f); for (int i = 0; i < n; ++i) { s = s + (QFEXPR); }   \
      g_sink = (double)s.f0; });                                                 \
    rows.push_back(r);                                                           \
  } while (0)

    std::vector<Row> rows;

    BENCH("add", "arith", xa[i] + xb[i], qa[i] + qb[i],
          da[i] + db[i], fa[i] + fb[i], ka[i] + kb[i]);
    BENCH("sub", "arith", xa[i] - xb[i], qa[i] - qb[i],
          da[i] - db[i], fa[i] - fb[i], ka[i] - kb[i]);
    BENCH("mul", "arith", xa[i] * xb[i], qa[i] * qb[i],
          da[i] * db[i], fa[i] * fb[i], ka[i] * kb[i]);
    BENCH("div", "arith", xa[i] / xb[i], qa[i] / qb[i],
          da[i] / db[i], fa[i] / fb[i], ka[i] / kb[i]);
    BENCH("sqrt", "arith", std::sqrt(xa[i]), sqrtq(qa[i]),
          Kokkos::sqrt(da[i]), Kokkos::sqrt(fa[i]), Kokkos::sqrt(ka[i]));
    BENCH("fma", "arith", std::fma(xa[i], xb[i], xa[i]), fmaq(qa[i], qb[i], qa[i]),
          ex::fma(da[i], db[i], da[i]), ex::fma(fa[i], fb[i], fa[i]),
          ex::fma(ka[i], kb[i], ka[i]));

    BENCH("exp", "trans", std::exp(xa[i] * 0.01), expq(qa[i] * (__float128)0.01),
          Kokkos::exp(da[i] * 0.01), Kokkos::exp(fa[i] * 0.01f),
          Kokkos::exp(ka[i] * 0.01f));
    BENCH("log", "trans", std::log(xa[i]), logq(qa[i]),
          Kokkos::log(da[i]), Kokkos::log(fa[i]), Kokkos::log(ka[i]));
    BENCH("sin", "trans", std::sin(xa[i]), sinq(qa[i]),
          Kokkos::sin(da[i]), Kokkos::sin(fa[i]), Kokkos::sin(ka[i]));
    BENCH("cos", "trans", std::cos(xa[i]), cosq(qa[i]),
          Kokkos::cos(da[i]), Kokkos::cos(fa[i]), Kokkos::cos(ka[i]));
    BENCH("atan", "trans", std::atan(xa[i]), atanq(qa[i]),
          Kokkos::atan(da[i]), Kokkos::atan(fa[i]), Kokkos::atan(ka[i]));
    BENCH("pow", "trans", std::pow(xa[i], 1.5), powq(qa[i], (__float128)1.5),
          Kokkos::pow(da[i], ex::DoubleDouble(1.5)),
          Kokkos::pow(fa[i], ex::FloatFloat(1.5)),
          Kokkos::pow(ka[i], ex::QuadFloat(1.5f)));
#undef BENCH

    if (rows.empty()) {
      std::fprintf(stderr, "error: no op matched '%s'\n", cfg.op.c_str());
      Kokkos::finalize();
      return 2;
    }

    std::printf("batch=%d  repeats=%d  seed=%llu  statistic=median  space=%s\n\n",
                cfg.batch, cfg.repeats, cfg.seed,
                Kokkos::DefaultExecutionSpace::name());
    std::printf("Cost relative to the incumbent at comparable precision.\n");
    std::printf("  FF (~14 digits) vs FP64 (~16)      "
                "| DD (~31) and QF (~29) vs __float128 (~34)\n\n");
    std::printf("%-6s %-6s | %10s %11s | %10s %11s %11s\n",
                "op", "class", "FP64 ms", "FF xFP64", "f128 ms", "DD xf128", "QF xf128");
    std::printf("-------------+-------------------------+"
                "-------------------------------------\n");
    for (const auto& r : rows)
      std::printf("%-6s %-6s | %10.3f %10.1fx | %10.3f %10.2fx %10.2fx\n",
                  r.op, r.cls, r.f64, r.ff / r.f64, r.q128, r.dd / r.q128, r.qf / r.q128);
    std::printf("\nA ratio below 1.00x means the backend is faster than the incumbent.\n");
  }
  Kokkos::finalize();
  return 0;
}
