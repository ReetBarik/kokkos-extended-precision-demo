// Probe: find which inputs cause the worst-accuracy results for a given op.
// Mirrors the demo's input generation (same seed, same draws), runs FF on host,
// prints the K worst elements with input bit patterns for reproducibility.
//
// Usage: ./build/probe_op exp           # default 1M elements, seed 12345
//        ./build/probe_op remainder
//        ./build/probe_op exp 100000 42 # batch=100000 seed=42

#include <Kokkos_Core.hpp>
extern "C" {
#include <quadmath.h>
}
#include <ff_math.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace ff = quad::ffun;

static uint64_t double_bits(double d) {
    uint64_t b; std::memcpy(&b, &d, sizeof(double)); return b;
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        std::string op = (argc > 1) ? argv[1] : "exp";
        int n          = (argc > 2) ? std::atoi(argv[2]) : 1'000'000;
        uint64_t seed  = (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 12345ULL;

        std::vector<double> ha(n), hb(n, 0.0);
        std::mt19937_64 gen(seed);
        if (op == "exp") {
            std::uniform_real_distribution<double> d(-80.0, 80.0);
            for (int i = 0; i < n; ++i) ha[i] = d(gen);
        } else if (op == "remainder") {
            std::uniform_real_distribution<double> da(0.1, 100.0), db(0.1, 10.0);
            for (int i = 0; i < n; ++i) { ha[i] = da(gen); hb[i] = db(gen); }
        } else {
            std::fprintf(stderr, "Unknown op: %s (supported: exp, remainder)\n", op.c_str());
            Kokkos::finalize();
            return 1;
        }

        // Run FF on host (KOKKOS_INLINE_FUNCTION resolves to plain functions on CPU).
        std::vector<ff::ffloat> rff((size_t)n);
        for (int i = 0; i < n; ++i) {
            ff::ffloat a(ha[i]), b(hb[i]);
            if      (op == "exp")       rff[i] = ff::exp(a);
            else if (op == "remainder") rff[i] = ff::remainder(a, b);
        }

        struct Bad {
            int idx;
            double a, b;
            float ff_hi, ff_lo;
            __float128 ref;
            double digits;
        };
        std::vector<Bad> all((size_t)n);
        for (int i = 0; i < n; ++i) {
            __float128 fa = (__float128)ha[i], fb = (__float128)hb[i];
            __float128 ref = 0.0q;
            if      (op == "exp")       ref = expq(fa);
            else if (op == "remainder") ref = remainderq(fa, fb);
            __float128 dev = (__float128)rff[i].hi + (__float128)rff[i].lo;
            double digits;
            if (isnanq(dev) || isnanq(ref)) digits = 0.0;
            else if (ref == (__float128)0.0) digits = (dev == (__float128)0.0) ? 14.0 : 0.0;
            else {
                __float128 rel = fabsq((dev - ref) / ref);
                digits = (rel == (__float128)0.0) ? 14.0 : -(double)log10q(rel);
                if (digits < 0)    digits = 0.0;
                if (digits > 14.0) digits = 14.0;
            }
            all[i] = {i, ha[i], hb[i], rff[i].hi, rff[i].lo, ref, digits};
        }

        std::sort(all.begin(), all.end(),
                  [](const Bad& x, const Bad& y) { return x.digits < y.digits; });

        std::printf("Worst 20 elements for op=%s (n=%d, seed=%llu):\n",
                    op.c_str(), n, (unsigned long long)seed);
        char ref_buf[128];
        for (int k = 0; k < 20 && k < n; ++k) {
            const Bad& bb = all[k];
            quadmath_snprintf(ref_buf, sizeof(ref_buf), "%.25Qe", bb.ref);
            std::printf("[%7d] digits=%6.3f  a=%.17g (0x%016llx)  b=%.17g\n"
                        "           ff=(%.9g, %.9g) -> %.17e   ref=%s\n",
                        bb.idx, bb.digits,
                        bb.a, (unsigned long long)double_bits(bb.a), bb.b,
                        bb.ff_hi, bb.ff_lo, (double)bb.ff_hi + (double)bb.ff_lo,
                        ref_buf);
        }
    }
    Kokkos::finalize();
    return 0;
}
