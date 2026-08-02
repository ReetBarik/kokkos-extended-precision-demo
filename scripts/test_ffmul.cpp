// Standalone host validator for ffmul (Dekker-splitting in FP32).
// Compares (hi + lo) against FP64 reference for many random inputs.
//
// Build: g++ -std=c++17 -O2 -ffloat-store scripts/test_ffmul.cpp -o scripts/test_ffmul
// Run:   ./scripts/test_ffmul

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <random>

struct ffloat { float hi, lo; };

static inline ffloat ffmul(ffloat a, ffloat b) {
    const float split = 8193.0f;
    float cona = a.hi * split, conb = b.hi * split;
    float a1 = cona - (cona - a.hi), b1 = conb - (conb - b.hi);
    float a2 = a.hi - a1,            b2 = b.hi - b1;
    float c11 = a.hi * b.hi;
    float c21 = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    float c2  = a.hi * b.lo + a.lo * b.hi;
    float t1  = c11 + c2;
    float e   = t1 - c11;
    float t2  = ((c2 - e) + (c11 - (t1 - e))) + c21 + a.lo * b.lo;
    float hi  = t1 + t2;
    float lo  = t2 - (hi - t1);
    return {hi, lo};
}

static inline ffloat ffmulff(float fa, float fb) {
    const float split = 8193.0f;
    float cona = fa * split, conb = fb * split;
    float a1 = cona - (cona - fa), b1 = conb - (conb - fb);
    float a2 = fa - a1,            b2 = fb - b1;
    float s1 = fa * fb;
    float s2 = (((a1*b1 - s1) + a1*b2) + a2*b1) + a2*b2;
    return {s1, s2};
}

int main() {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> uni(-1e6, 1e6);

    // Test 1: ffmulff(float, float) must be EXACT (no rounding error)
    int exact_fail = 0;
    int N = 200000;
    for (int i = 0; i < N; ++i) {
        float x = (float)uni(rng);
        float y = (float)uni(rng);
        ffloat r = ffmulff(x, y);
        // Sum r.hi + r.lo must equal exactly (double)x * (double)y
        double expected = (double)x * (double)y;
        double got      = (double)r.hi + (double)r.lo;
        if (expected != got) {
            if (exact_fail < 5) {
                std::printf("EXACT FAIL: %.9g * %.9g  expected=%.18g got=%.18g\n",
                            x, y, expected, got);
            }
            ++exact_fail;
        }
    }
    std::printf("ffmulff exactness: %d/%d failures\n", exact_fail, N);

    // Test 2: ffmul(ffloat, ffloat) where each input is the FF representation of a random FP64
    // Expected relative error ~ 2^-48 ~ 3.5e-15.
    auto fp64_to_ff = [](double v) -> ffloat {
        float h = (float)v;
        float l = (float)(v - (double)h);
        return {h, l};
    };

    double max_rel_err = 0.0;
    double sum_rel_err = 0.0;
    int sample_fail = 0;
    for (int i = 0; i < N; ++i) {
        double a = uni(rng);
        double b = uni(rng);
        ffloat fa = fp64_to_ff(a), fb = fp64_to_ff(b);
        ffloat r = ffmul(fa, fb);
        double got      = (double)r.hi + (double)r.lo;
        double expected = a * b;
        double rel_err  = std::fabs((got - expected) / expected);
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        sum_rel_err += rel_err;
        if (rel_err > 1.0e-12) ++sample_fail;
    }
    std::printf("ffmul rel-err: max=%.3e  mean=%.3e  N=%d  outliers(>1e-12)=%d\n",
                max_rel_err, sum_rel_err/N, N, sample_fail);

    // Test 3: ffmul on specific tight test cases
    auto check = [&](const char* label, double a, double b) {
        ffloat fa = fp64_to_ff(a), fb = fp64_to_ff(b);
        ffloat r  = ffmul(fa, fb);
        double got = (double)r.hi + (double)r.lo;
        double expected = a * b;
        double rel = std::fabs((got - expected) / expected);
        std::printf("  %-25s  expected=%.20g  got=%.20g  rel=%.3e\n",
                    label, expected, got, rel);
    };
    std::printf("Specific cases:\n");
    check("pi * pi",        3.141592653589793, 3.141592653589793);
    check("e * e",          2.718281828459045, 2.718281828459045);
    check("sqrt(2) * sqrt(2)", 1.4142135623730951, 1.4142135623730951);
    check("123.456 * 789.012", 123.456, 789.012);
    check("1.0001 * 0.9999", 1.0001, 0.9999);
    check("tiny * huge", 1.0e-15, 1.0e15);

    std::printf("\nAcceptance: max_rel_err < 5e-15 ? %s\n", (max_rel_err < 5e-15) ? "YES" : "NO");
    return (exact_fail == 0 && max_rel_err < 5e-15) ? 0 : 1;
}
