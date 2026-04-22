// Kokkos CUDA FP128 extended-precision demo — complex ops.
// Benchmarks quad_complex (FP128) and Kokkos::complex<double> against
// the host __complex128 quadmath reference. Prints three tables:
//   1. Timing
//   2. Real-part accuracy (precise digits)
//   3. Imaginary-part accuracy (precise digits)

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

extern "C" {
#include <quadmath.h>
}

#include <NVIDIA_emulated_quad/quad_math.hpp>
#include <NVIDIA_emulated_quad/quad_complex.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int      kWarmupRuns     = 2;
constexpr int      kDefaultRepeats = 5;
constexpr uint64_t kDefaultSeed    = 12345ULL;

// clang-format off
enum class Op {
  Add, Sub, Mul, Div,
  Abs, Conj, Sqrt, Exp, Log, Log10,
  Sin, Cos, Tan, Asin, Acos, Atan,
  Sinh, Cosh, Tanh, Asinh, Acosh, Atanh,
  Pow, Polar,
};

static const Op kAllOps[] = {
  Op::Add, Op::Sub, Op::Mul, Op::Div,
  Op::Abs, Op::Conj, Op::Sqrt, Op::Exp, Op::Log, Op::Log10,
  Op::Sin, Op::Cos, Op::Tan, Op::Asin, Op::Acos, Op::Atan,
  Op::Sinh, Op::Cosh, Op::Tanh, Op::Asinh, Op::Acosh, Op::Atanh,
  Op::Pow, Op::Polar,
};
// clang-format on

struct Config {
  Op       op      = Op::Add;
  bool     all_ops = false;
  int      batch   = 1'000'000;
  int      repeats = kDefaultRepeats;
  uint64_t seed    = kDefaultSeed;
};

bool parse_op(const std::string& s, Op& out) {
  // clang-format off
  if (s=="add")   {out=Op::Add;   return true;}
  if (s=="sub")   {out=Op::Sub;   return true;}
  if (s=="mul")   {out=Op::Mul;   return true;}
  if (s=="div")   {out=Op::Div;   return true;}
  if (s=="abs")   {out=Op::Abs;   return true;}
  if (s=="conj")  {out=Op::Conj;  return true;}
  if (s=="sqrt")  {out=Op::Sqrt;  return true;}
  if (s=="exp")   {out=Op::Exp;   return true;}
  if (s=="log")   {out=Op::Log;   return true;}
  if (s=="log10") {out=Op::Log10; return true;}
  if (s=="sin")   {out=Op::Sin;   return true;}
  if (s=="cos")   {out=Op::Cos;   return true;}
  if (s=="tan")   {out=Op::Tan;   return true;}
  if (s=="asin")  {out=Op::Asin;  return true;}
  if (s=="acos")  {out=Op::Acos;  return true;}
  if (s=="atan")  {out=Op::Atan;  return true;}
  if (s=="sinh")  {out=Op::Sinh;  return true;}
  if (s=="cosh")  {out=Op::Cosh;  return true;}
  if (s=="tanh")  {out=Op::Tanh;  return true;}
  if (s=="asinh") {out=Op::Asinh; return true;}
  if (s=="acosh") {out=Op::Acosh; return true;}
  if (s=="atanh") {out=Op::Atanh; return true;}
  if (s=="pow")   {out=Op::Pow;   return true;}
  if (s=="polar") {out=Op::Polar; return true;}
  // clang-format on
  return false;
}

const char* op_name(Op op) {
  switch (op) {
    case Op::Add:   return "add";
    case Op::Sub:   return "sub";
    case Op::Mul:   return "mul";
    case Op::Div:   return "div";
    case Op::Abs:   return "abs";
    case Op::Conj:  return "conj";
    case Op::Sqrt:  return "sqrt";
    case Op::Exp:   return "exp";
    case Op::Log:   return "log";
    case Op::Log10: return "log10";
    case Op::Sin:   return "sin";
    case Op::Cos:   return "cos";
    case Op::Tan:   return "tan";
    case Op::Asin:  return "asin";
    case Op::Acos:  return "acos";
    case Op::Atan:  return "atan";
    case Op::Sinh:  return "sinh";
    case Op::Cosh:  return "cosh";
    case Op::Tanh:  return "tanh";
    case Op::Asinh: return "asinh";
    case Op::Acosh: return "acosh";
    case Op::Atanh: return "atanh";
    case Op::Pow:   return "pow";
    case Op::Polar: return "polar";
  }
  return "?";
}

void print_usage(const char* argv0) {
  std::cerr
    << "Usage: " << argv0 << " [--op <name>] [--batch N] [--repeats N] [--seed N]\n"
    << "  Omit --op to run all operations and print complete tables.\n"
    << "  Operations: add sub mul div abs conj sqrt exp log log10\n"
    << "              sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh pow polar\n"
    << "  Defaults: batch=1000000 repeats=" << kDefaultRepeats << " seed=" << kDefaultSeed << "\n";
}

bool parse_args(int argc, char** argv, Config& cfg) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--help" || a == "-h") return false;
    auto need = [&](const char* w) -> const char* {
      if (i+1>=argc){std::cerr<<"Missing value after "<<w<<"\n";return nullptr;}
      return argv[++i];
    };
    if (a=="--op") {
      const char* v=need("--op"); if(!v) return false;
      if(!parse_op(v,cfg.op)){std::cerr<<"Unknown op: "<<v<<"\n";return false;}
      cfg.all_ops=false;
    } else if (a=="--batch") {
      const char* v=need("--batch"); if(!v) return false;
      cfg.batch=std::atoi(v);
      if(cfg.batch<=0){std::cerr<<"Invalid --batch\n";return false;}
    } else if (a=="--repeats") {
      const char* v=need("--repeats"); if(!v) return false;
      cfg.repeats=std::atoi(v);
      if(cfg.repeats<=0){std::cerr<<"Invalid --repeats\n";return false;}
    } else if (a=="--seed") {
      const char* v=need("--seed"); if(!v) return false;
      cfg.seed=static_cast<uint64_t>(std::strtoull(v,nullptr,10));
    } else {
      std::cerr<<"Unknown argument: "<<a<<"\n"; return false;
    }
  }
  return true;
}

// ---- Input generation -------------------------------------------------------

void fill_inputs(Op op, double* ha_re, double* ha_im,
                         double* hb_re, double* hb_im, int n, uint64_t seed) {
  std::mt19937_64 gen(seed);
  constexpr double pi = 3.14159265358979323846;

  auto fill_a = [&](double lo_re, double hi_re, double lo_im, double hi_im) {
    std::uniform_real_distribution<double> dr(lo_re,hi_re), di(lo_im,hi_im);
    for (int i=0;i<n;++i){ha_re[i]=dr(gen);ha_im[i]=di(gen);hb_re[i]=0;hb_im[i]=0;}
  };
  auto fill_ab = [&](double lo_re, double hi_re, double lo_im, double hi_im) {
    std::uniform_real_distribution<double> dr(lo_re,hi_re), di(lo_im,hi_im);
    for (int i=0;i<n;++i){ha_re[i]=dr(gen);ha_im[i]=di(gen);hb_re[i]=dr(gen);hb_im[i]=di(gen);}
  };

  switch (op) {
    case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: fill_ab(0.1,10.0,0.1,10.0); break;
    case Op::Abs: case Op::Conj: fill_a(-10,10,-10,10); break;
    case Op::Sqrt:  fill_a(-10,10,-10,10); break;
    case Op::Exp:   fill_a(-10,10,-pi,pi); break;
    case Op::Log: case Op::Log10: fill_a(0.1,10,-10,10); break;
    case Op::Sin: case Op::Cos: case Op::Tan: fill_a(-pi,pi,-2,2); break;
    case Op::Asin: case Op::Acos: fill_a(-1,1,-1,1); break;
    case Op::Atan:  fill_a(-10,10,-0.9,0.9); break;
    case Op::Sinh: case Op::Cosh: case Op::Tanh: fill_a(-5,5,-pi,pi); break;
    case Op::Asinh: fill_a(-10,10,-10,10); break;
    case Op::Acosh: fill_a(0,10,-5,5); break;
    case Op::Atanh: fill_a(-0.9,0.9,-0.9,0.9); break;
    case Op::Pow: {
      std::uniform_real_distribution<double> dbre(0.1,10),dbim(-5,5),dere(0,3),deim(-1,1);
      for (int i=0;i<n;++i){ha_re[i]=dbre(gen);ha_im[i]=dbim(gen);hb_re[i]=dere(gen);hb_im[i]=deim(gen);}
      break;
    }
    case Op::Polar: {
      std::uniform_real_distribution<double> dr(0.01,100), dth(-pi,pi);
      for (int i=0;i<n;++i){ha_re[i]=dr(gen);ha_im[i]=dth(gen);hb_re[i]=0;hb_im[i]=0;}
      break;
    }
  }
}

// ---- Host quadmath reference ------------------------------------------------

void host_quadmath_reference(Op op,
                             const double* ha_re, const double* ha_im,
                             const double* hb_re, const double* hb_im,
                             __float128* out_re, __float128* out_im, int n) {
  for (int i = 0; i < n; ++i) {
    __complex128 za; __real__ za=(__float128)ha_re[i]; __imag__ za=(__float128)ha_im[i];
    __complex128 zb; __real__ zb=(__float128)hb_re[i]; __imag__ zb=(__float128)hb_im[i];
    __complex128 res = 0.0q;
    switch (op) {
      case Op::Add:   res = za + zb;       break;
      case Op::Sub:   res = za - zb;       break;
      case Op::Mul:   res = za * zb;       break;
      case Op::Div:   res = za / zb;       break;
      case Op::Abs:   out_re[i]=cabsq(za); out_im[i]=0.0q; continue;
      case Op::Conj:  res = conjq(za);     break;
      case Op::Sqrt:  res = csqrtq(za);    break;
      case Op::Exp:   res = cexpq(za);     break;
      case Op::Log:   res = clogq(za);     break;
      case Op::Log10: res = clog10q(za);   break;
      case Op::Sin:   res = csinq(za);     break;
      case Op::Cos:   res = ccosq(za);     break;
      case Op::Tan:   res = ctanq(za);     break;
      case Op::Asin:  res = casinq(za);    break;
      case Op::Acos:  res = cacosq(za);    break;
      case Op::Atan:  res = catanq(za);    break;
      case Op::Sinh:  res = csinhq(za);    break;
      case Op::Cosh:  res = ccoshq(za);    break;
      case Op::Tanh:  res = ctanhq(za);    break;
      case Op::Asinh: res = casinhq(za);   break;
      case Op::Acosh: res = cacoshq(za);   break;
      case Op::Atanh: res = catanhq(za);   break;
      case Op::Pow:   res = cpowq(za,zb);  break;
      case Op::Polar: {
        __float128 r=(__float128)ha_re[i], th=(__float128)ha_im[i];
        out_re[i]=r*cosq(th); out_im[i]=r*sinq(th); continue;
      }
    }
    out_re[i]=crealq(res); out_im[i]=cimagq(res);
  }
}

// ---- Timing ----------------------------------------------------------------

struct TimeStats { double min_s=0, max_s=0, median_s=0, mean_s=0; };

TimeStats summarize_times(std::vector<double> t) {
  if (t.empty()) return {};
  std::sort(t.begin(), t.end());
  TimeStats s;
  s.min_s    = t.front();
  s.max_s    = t.back();
  size_t n   = t.size();
  s.median_s = (n%2==1) ? t[n/2] : 0.5*(t[n/2-1]+t[n/2]);
  s.mean_s   = std::accumulate(t.begin(),t.end(),0.0)/(double)n;
  return s;
}

using wall_clock = std::chrono::high_resolution_clock;

template <typename Launch>
TimeStats time_kernel_fence(int repeats, Launch&& launch) {
  for (int w=0;w<kWarmupRuns;++w){launch();Kokkos::fence();}
  std::vector<double> times; times.reserve((size_t)repeats);
  for (int r=0;r<repeats;++r) {
    auto t0=wall_clock::now(); launch(); Kokkos::fence();
    times.push_back(std::chrono::duration<double>(wall_clock::now()-t0).count());
  }
  return summarize_times(std::move(times));
}

// ---- Accuracy --------------------------------------------------------------

struct AccStats { double min_d=33, max_d=0, mean_d=0, median_d=0; };

static double element_digits(__float128 dev, __float128 ref) {
  if (isnanq(dev)||isnanq(ref)) return 0.0;
  if (isinfq(ref)) return (isinfq(dev)&&(dev>0)==(ref>0))?33.0:0.0;
  if (ref==(__float128)0.0) return (dev==(__float128)0.0)?33.0:0.0;
  __float128 rel=fabsq((dev-ref)/ref);
  if (rel==(__float128)0.0) return 33.0;
  double d=-(double)log10q(rel);
  return d<0.0?0.0:(d>33.0?33.0:d);
}

AccStats compute_acc_fp128(const quad::cuda_fp128::fp128_t* dev, const __float128* ref, int n) {
  std::vector<double> digs((size_t)n);
  for (int i=0;i<n;++i) digs[i]=element_digits(static_cast<__float128>(dev[i].value),ref[i]);
  std::sort(digs.begin(),digs.end());
  AccStats s;
  s.min_d=digs.front(); s.max_d=digs.back();
  s.mean_d=std::accumulate(digs.begin(),digs.end(),0.0)/(double)n;
  size_t m=digs.size();
  s.median_d=(m%2==1)?digs[m/2]:0.5*(digs[m/2-1]+digs[m/2]);
  return s;
}

AccStats compute_acc_dbl(const double* dev, const __float128* ref, int n) {
  std::vector<double> digs((size_t)n);
  for (int i=0;i<n;++i) digs[i]=element_digits((__float128)dev[i],ref[i]);
  std::sort(digs.begin(),digs.end());
  AccStats s;
  s.min_d=digs.front(); s.max_d=digs.back();
  s.mean_d=std::accumulate(digs.begin(),digs.end(),0.0)/(double)n;
  size_t m=digs.size();
  s.median_d=(m%2==1)?digs[m/2]:0.5*(digs[m/2-1]+digs[m/2]);
  return s;
}

// ---- Per-op result ---------------------------------------------------------

struct ComplexOpResult {
  Op        op;
  TimeStats fp128_timing, dbl_timing;
  AccStats  fp128_re, fp128_im;
  AccStats  dbl_re,   dbl_im;
};

// ---- Execution space -------------------------------------------------------

using exec_space = Kokkos::DefaultExecutionSpace;
using policy_1d  = Kokkos::RangePolicy<exec_space>;
using vqc        = Kokkos::View<quad::cuda_fp128::quad_complex*,        Kokkos::LayoutRight, exec_space>;
using vdc        = Kokkos::View<Kokkos::complex<double>*, Kokkos::LayoutRight, exec_space>;
using v128       = Kokkos::View<quad::cuda_fp128::fp128_t*,             Kokkos::LayoutRight, exec_space>;

ComplexOpResult run_op(Op op, const Config& cfg) {
  const int n = cfg.batch;

  std::vector<double> ha_re(n),ha_im(n),hb_re(n),hb_im(n);
  fill_inputs(op, ha_re.data(),ha_im.data(), hb_re.data(),hb_im.data(), n, cfg.seed);

  std::vector<__float128> href_re(n), href_im(n);
  host_quadmath_reference(op, ha_re.data(),ha_im.data(), hb_re.data(),hb_im.data(),
                          href_re.data(), href_im.data(), n);

  vqc  qa("qa",n), qb("qb",n), qr("qr",n);
  vdc  da("da",n), db("db",n), dr("dr",n);
  v128 qra("qra",n), qrb("qrb",n);

  {
    auto mqa=Kokkos::create_mirror_view(qa), mqb=Kokkos::create_mirror_view(qb);
    auto mda=Kokkos::create_mirror_view(da), mdb=Kokkos::create_mirror_view(db);
    auto mqra=Kokkos::create_mirror_view(qra), mqrb=Kokkos::create_mirror_view(qrb);
    for (int i=0;i<n;++i) {
      mqa(i)=quad::cuda_fp128::quad_complex(quad::cuda_fp128::fp128_t((__float128)ha_re[i]),quad::cuda_fp128::fp128_t((__float128)ha_im[i]));
      mqb(i)=quad::cuda_fp128::quad_complex(quad::cuda_fp128::fp128_t((__float128)hb_re[i]),quad::cuda_fp128::fp128_t((__float128)hb_im[i]));
      mda(i)=Kokkos::complex<double>(ha_re[i],ha_im[i]);
      mdb(i)=Kokkos::complex<double>(hb_re[i],hb_im[i]);
      mqra(i)=quad::cuda_fp128::fp128_t((__float128)ha_re[i]);
      mqrb(i)=quad::cuda_fp128::fp128_t((__float128)ha_im[i]);
    }
    Kokkos::deep_copy(qa,mqa); Kokkos::deep_copy(qb,mqb);
    Kokkos::deep_copy(da,mda); Kokkos::deep_copy(db,mdb);
    Kokkos::deep_copy(qra,mqra); Kokkos::deep_copy(qrb,mqrb);
  }

  policy_1d pol(0,n);
  TimeStats st_fp128, st_dbl;

  // ---- FP128 kernels -------------------------------------------------------
  switch (op) {
    case Op::Add:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_add",pol,KOKKOS_LAMBDA(int i){qr(i)=qa(i)+qb(i);});}); break;
    case Op::Sub:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_sub",pol,KOKKOS_LAMBDA(int i){qr(i)=qa(i)-qb(i);});}); break;
    case Op::Mul:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_mul",pol,KOKKOS_LAMBDA(int i){qr(i)=qa(i)*qb(i);});}); break;
    case Op::Div:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_div",pol,KOKKOS_LAMBDA(int i){qr(i)=qa(i)/qb(i);});}); break;
    case Op::Abs:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_abs",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::quad_complex(quad::cuda_fp128::abs(qa(i)),quad::cuda_fp128::fp128_t(0.0));});}); break;
    case Op::Conj:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_conj",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::conj(qa(i));});}); break;
    case Op::Sqrt:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_sqrt",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::sqrt(qa(i));});}); break;
    case Op::Exp:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_exp",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::exp(qa(i));});}); break;
    case Op::Log:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_log",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::log(qa(i));});}); break;
    case Op::Log10:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_log10",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::log10(qa(i));});}); break;
    case Op::Sin:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_sin",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::sin(qa(i));});}); break;
    case Op::Cos:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_cos",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::cos(qa(i));});}); break;
    case Op::Tan:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_tan",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::tan(qa(i));});}); break;
    case Op::Asin:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_asin",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::asin(qa(i));});}); break;
    case Op::Acos:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_acos",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::acos(qa(i));});}); break;
    case Op::Atan:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_atan",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::atan(qa(i));});}); break;
    case Op::Sinh:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_sinh",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::sinh(qa(i));});}); break;
    case Op::Cosh:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_cosh",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::cosh(qa(i));});}); break;
    case Op::Tanh:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_tanh",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::tanh(qa(i));});}); break;
    case Op::Asinh:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_asinh",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::asinh(qa(i));});}); break;
    case Op::Acosh:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_acosh",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::acosh(qa(i));});}); break;
    case Op::Atanh:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_atanh",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::atanh(qa(i));});}); break;
    case Op::Pow:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_pow",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::pow(qa(i),qb(i));});}); break;
    case Op::Polar:
      st_fp128=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("qc_polar",pol,KOKKOS_LAMBDA(int i){qr(i)=quad::cuda_fp128::polar(qra(i),qrb(i));});}); break;
  }

  // ---- double complex kernels ----------------------------------------------
  switch (op) {
    case Op::Add:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_add",pol,KOKKOS_LAMBDA(int i){dr(i)=da(i)+db(i);});}); break;
    case Op::Sub:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_sub",pol,KOKKOS_LAMBDA(int i){dr(i)=da(i)-db(i);});}); break;
    case Op::Mul:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_mul",pol,KOKKOS_LAMBDA(int i){dr(i)=da(i)*db(i);});}); break;
    case Op::Div:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_div",pol,KOKKOS_LAMBDA(int i){dr(i)=da(i)/db(i);});}); break;
    case Op::Abs:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_abs",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::complex<double>(Kokkos::abs(da(i)),0.0);});}); break;
    case Op::Conj:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_conj",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::conj(da(i));});}); break;
    case Op::Sqrt:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_sqrt",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::sqrt(da(i));});}); break;
    case Op::Exp:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_exp",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::exp(da(i));});}); break;
    case Op::Log:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_log",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::log(da(i));});}); break;
    case Op::Log10:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_log10",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::log10(da(i));});}); break;
    case Op::Sin:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_sin",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::sin(da(i));});}); break;
    case Op::Cos:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_cos",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::cos(da(i));});}); break;
    case Op::Tan:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_tan",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::tan(da(i));});}); break;
    case Op::Asin:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_asin",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::asin(da(i));});}); break;
    case Op::Acos:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_acos",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::acos(da(i));});}); break;
    case Op::Atan:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_atan",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::atan(da(i));});}); break;
    case Op::Sinh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_sinh",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::sinh(da(i));});}); break;
    case Op::Cosh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_cosh",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::cosh(da(i));});}); break;
    case Op::Tanh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_tanh",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::tanh(da(i));});}); break;
    case Op::Asinh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_asinh",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::asinh(da(i));});}); break;
    case Op::Acosh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_acosh",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::acosh(da(i));});}); break;
    case Op::Atanh:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_atanh",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::atanh(da(i));});}); break;
    case Op::Pow:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_pow",pol,KOKKOS_LAMBDA(int i){dr(i)=Kokkos::pow(da(i),db(i));});}); break;
    case Op::Polar:
      st_dbl=time_kernel_fence(cfg.repeats,[&](){Kokkos::parallel_for("dc_polar",pol,KOKKOS_LAMBDA(int i){
        dr(i)=Kokkos::complex<double>(da(i).real()*Kokkos::cos(da(i).imag()),
                                      da(i).real()*Kokkos::sin(da(i).imag()));
      });}); break;
  }

  // ---- Download and compute accuracy ---------------------------------------
  auto mqr=Kokkos::create_mirror_view(qr); Kokkos::deep_copy(mqr,qr);
  auto mdr=Kokkos::create_mirror_view(dr); Kokkos::deep_copy(mdr,dr);

  std::vector<quad::cuda_fp128::fp128_t> qr_re(n), qr_im(n);
  std::vector<double>      dr_re(n), dr_im(n);
  for (int i=0;i<n;++i) {
    qr_re[i]=mqr(i).re; qr_im[i]=mqr(i).im;
    dr_re[i]=mdr(i).real(); dr_im[i]=mdr(i).imag();
  }

  return { op, st_fp128, st_dbl,
           compute_acc_fp128(qr_re.data(), href_re.data(), n),
           compute_acc_fp128(qr_im.data(), href_im.data(), n),
           compute_acc_dbl(dr_re.data(), href_re.data(), n),
           compute_acc_dbl(dr_im.data(), href_im.data(), n) };
}

// ---- Table printing --------------------------------------------------------
// Layout matches user's requested format:
//   OP (10) | backend1: Time(4×kTW) | Accuracy(4×kAW) | backend2: same |

static constexpr int kOpW = 12;
static constexpr int kTW  =  9;  // time column (ms, fixed 4dp)
static constexpr int kAW  =  7;  // accuracy column (digits, fixed 2dp)
static constexpr int kTSec = 4*kTW + 3;  // 39
static constexpr int kASec = 4*kAW + 3;  // 31
static constexpr int kBkW  = kTSec + 1 + kASec;  // 71

static std::string dashes(int n) { return std::string((size_t)n,'-'); }

static std::string center(const std::string& s, int w) {
  int pad=w-(int)s.size(), lp=pad/2, rp=pad-lp;
  return std::string((size_t)lp,' ')+s+std::string((size_t)rp,' ');
}

static void print_sep() {
  std::cout
    << '-' << dashes(kOpW) << "-+"
    << dashes(kTSec) << "+" << dashes(kASec) << "+"
    << dashes(kTSec) << "+" << dashes(kASec) << "+\n";
}

static void print_header(const char* lbl1, const char* lbl2, const char* acc_label) {
  using std::cout;
  cout << ' ' << std::string(kOpW,' ') << " |"
       << center(lbl1, kBkW) << "|"
       << center(lbl2, kBkW) << "|\n";
  cout << ' ' << std::string(kOpW,' ') << " |"
       << center("Time (ms)", kTSec) << "|" << center(acc_label, kASec) << "|"
       << center("Time (ms)", kTSec) << "|" << center(acc_label, kASec) << "|\n";
  print_sep();
  cout << ' ' << std::left << std::setw(kOpW) << "" << " |"
       << center("Min",kTW) << "|" << center("Max",kTW) << "|"
       << center("Med",kTW) << "|" << center("Mean",kTW) << "|"
       << center("Min",kAW) << "|" << center("Max",kAW) << "|"
       << center("Med",kAW) << "|" << center("Mean",kAW) << "|"
       << center("Min",kTW) << "|" << center("Max",kTW) << "|"
       << center("Med",kTW) << "|" << center("Mean",kTW) << "|"
       << center("Min",kAW) << "|" << center("Max",kAW) << "|"
       << center("Med",kAW) << "|" << center("Mean",kAW) << "|\n";
  print_sep();
}

static void print_row(const char* name,
                      const TimeStats& t1, const AccStats& a1,
                      const TimeStats& t2, const AccStats& a2) {
  using std::cout; using std::setw; using std::right; using std::fixed; using std::setprecision;
  auto T=[](double s){return s*1000.0;};
  cout << ' ' << std::left << std::setw(kOpW) << name << " |"
       << right << fixed << setprecision(4)
       << setw(kTW)<<T(t1.min_s)   <<"|"<< setw(kTW)<<T(t1.max_s)    <<"|"
       << setw(kTW)<<T(t1.median_s)<<"|"<< setw(kTW)<<T(t1.mean_s)   <<"|"
       << setprecision(2)
       << setw(kAW)<<a1.min_d      <<"|"<< setw(kAW)<<a1.max_d       <<"|"
       << setw(kAW)<<a1.median_d   <<"|"<< setw(kAW)<<a1.mean_d      <<"|"
       << setprecision(4)
       << setw(kTW)<<T(t2.min_s)   <<"|"<< setw(kTW)<<T(t2.max_s)    <<"|"
       << setw(kTW)<<T(t2.median_s)<<"|"<< setw(kTW)<<T(t2.mean_s)   <<"|"
       << setprecision(2)
       << setw(kAW)<<a2.min_d      <<"|"<< setw(kAW)<<a2.max_d       <<"|"
       << setw(kAW)<<a2.median_d   <<"|"<< setw(kAW)<<a2.mean_d      <<"|\n";
}

static void print_inner_sep() {
  std::cout << ' ' << std::string(kOpW, ' ') << " +"
            << dashes(kTSec) << "+" << dashes(kASec) << "+"
            << dashes(kTSec) << "+" << dashes(kASec) << "+\n";
}

static void print_complex_op_rows(const ComplexOpResult& r) {
  std::string re_n = std::string(op_name(r.op)) + " (real)";
  std::string im_n = std::string(op_name(r.op)) + " (imag)";
  print_row(re_n.c_str(), r.fp128_timing, r.fp128_re, r.dbl_timing, r.dbl_re);
  print_inner_sep();
  print_row(im_n.c_str(), r.fp128_timing, r.fp128_im, r.dbl_timing, r.dbl_im);
  print_sep();
}


}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  cfg.all_ops = true;
  for (int i=1;i<argc;++i)
    if (std::string(argv[i])=="--op") { cfg.all_ops=false; break; }

  if (!parse_args(argc, argv, cfg)) { print_usage(argv[0]); return 1; }

  Kokkos::initialize(argc, argv);
  {
    std::cout << "\nbatch=" << cfg.batch << "  repeats=" << cfg.repeats
              << "  seed=" << cfg.seed << "  warmup=" << kWarmupRuns
              << "  timing=kernel+fence\n\n";
    print_header("CUDA Emulated FP128", "CUDA FP64", "Accuracy (dig)");
    if (cfg.all_ops) {
      for (Op op : kAllOps) {
        print_complex_op_rows(run_op(op, cfg));
      }
    } else {
      print_complex_op_rows(run_op(cfg.op, cfg));
    }
    std::cout << "\n";
  }
  Kokkos::finalize();
  return 0;
}
