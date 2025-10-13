#include <arm_fp16.h>
#include <arm_sve.h>
#include <math.h>  // for isnan if needed
#include <stddef.h>
#include "mlasi_sve.h"
using _mlas_fp16_ = uint16_t;
struct MlasTanhConstants_fp16_scalar {
    __fp16 LowerRange;
    __fp16 UpperRange;
    __fp16 alpha_7;
    __fp16 alpha_5;
    __fp16 alpha_3;
    __fp16 alpha_1;
    __fp16 beta_6;
    __fp16 beta_4;
    __fp16 beta_2;
    __fp16 beta_0;
};
// Your original scalar constants (replace these with your actual values)
constexpr MlasTanhConstants_fp16_scalar TanhConstantsFp16 = {
    -3.515625f,               // LowerRange
    3.515625f,                // UpperRange
    5.960464477539063e-08f,   // alpha_7 (replace with real constant)
    1.4841556549072266e-05f,  // alpha_5
    0.000637054443359375f,    // alpha_3
    0.004894256591796875f,    // alpha_1
    1.1920928955078125e-06f,  // beta_6
    0.00011855363845825195f,  // beta_4
    0.0022678375244140625f,   // beta_2
    0.004894256591796875f     // beta_0
};
// Tanh vector approximation on one SVE vector chunk
static inline svfloat16_t
Tanh_Vector_SVE_fp16(svfloat16_t x, svbool_t pg)
{
    svfloat16_t g_LowerRange_vec = svdup_f16(TanhConstantsFp16.LowerRange);
    svfloat16_t g_UpperRange_vec = svdup_f16(TanhConstantsFp16.UpperRange);
    svfloat16_t g_alpha_7_vec = svdup_f16(TanhConstantsFp16.alpha_7);
    svfloat16_t g_alpha_5_vec = svdup_f16(TanhConstantsFp16.alpha_5);
    svfloat16_t g_alpha_3_vec = svdup_f16(TanhConstantsFp16.alpha_3);
    svfloat16_t g_alpha_1_vec = svdup_f16(TanhConstantsFp16.alpha_1);
    svfloat16_t g_beta_6_vec = svdup_f16(TanhConstantsFp16.beta_6);
    svfloat16_t g_beta_4_vec = svdup_f16(TanhConstantsFp16.beta_4);
    svfloat16_t g_beta_2_vec = svdup_f16(TanhConstantsFp16.beta_2);
    svfloat16_t g_beta_0_vec = svdup_f16(TanhConstantsFp16.beta_0);
    // Clamp x between LowerRange and UpperRange
    x = svmin_f16_m(pg, x, g_UpperRange_vec);
    x = svmax_f16_m(pg, x, g_LowerRange_vec);
    svfloat16_t x2 = svmul_f16_m(pg, x, x);
    // numerator p = (((α7 * x2 + α5) * x2 + α3) * x2 + α1) * x
    svfloat16_t p = svmla_f16_m(pg, g_alpha_5_vec, g_alpha_7_vec, x2);  // α7*x2 + α5
    p = svmla_f16_m(pg, g_alpha_3_vec, p, x2);
    p = svmla_f16_m(pg, g_alpha_1_vec, p, x2);
    p = svmul_f16_m(pg, p, x);
    // denominator q = (((β6 * x2 + β4) * x2 + β2) * x2 + β0)
    svfloat16_t q = svmla_f16_m(pg, g_beta_4_vec, g_beta_6_vec, x2);
    q = svmla_f16_m(pg, g_beta_2_vec, q, x2);
    q = svmla_f16_m(pg, g_beta_0_vec, q, x2);
    // result = p / q
    svfloat16_t res = svdiv_f16_m(pg, p, q);
    return res;
}
// Main tanh kernel applying SVE fp16 vectorization
void
MlasTanhKernelFp16_SVE(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N)
{
    size_t offset = 0;
    const auto* input = reinterpret_cast<const _mlas_fp16_*>(Input);
    auto* output = reinterpret_cast<_mlas_fp16_*>(Output);
    while (offset < N) {
        // Predicate mask for valid elements
        svbool_t pg = svwhilelt_b16(offset, N);
        // Load input vector slice
        svfloat16_t x = svreinterpret_f16_u16(svld1_u16(pg, &input[offset]));
        // Compute tanh on vector slice
        svfloat16_t y = Tanh_Vector_SVE_fp16(x, pg);
        svst1_u16(pg, &output[offset], svreinterpret_u16_f16(y));
        // Increment offset by vector length in 16-bit elements
        // svcntw() returns number of 32-bit lanes, so multiply by 2 for 16-bit lanes
        offset += svcntw() * 2;
    }
}
static inline svfloat16_t
exp_neg_rational_approx_f16(svbool_t pg, svfloat16_t x)
{
    // Clamp x to avoid overflow
    svfloat16_t max_x = svdup_f16(6.0f);
    x = svmin_f16_m(pg, x, max_x);
    const float16_t c0 = 1.330f;
    const float16_t c1 = -0.390f;
    const float16_t c2 = 0.0288f;
    const float16_t d0 = 1.338f;
    const float16_t d1 = 0.848f;
    const float16_t d2 = 0.467f;
    // Broadcast coefficients
    svfloat16_t c0v = svdup_f16(c0);
    svfloat16_t c1v = svdup_f16(c1);
    svfloat16_t c2v = svdup_f16(c2);
    svfloat16_t d0v = svdup_f16(d0);
    svfloat16_t d1v = svdup_f16(d1);
    svfloat16_t d2v = svdup_f16(d2);
    // Powers of x
    svfloat16_t x2 = svmul_f16_m(pg, x, x);
    // numerator = c0 + c1*x + c2*x²
    svfloat16_t num = svmla_f16_m(pg, c0v, c1v, x);
    num = svmla_f16_m(pg, num, c2v, x2);
    // denominator = d0 + d1*x + d2*x²
    svfloat16_t den = svmla_f16_m(pg, d0v, d1v, x);
    den = svmla_f16_m(pg, den, d2v, x2);
    // Reciprocal approximation via Newton-Raphson
    svfloat16_t recip = svrecpe_f16(den);
    recip = svmul_f16_m(pg, recip, svrecps_f16(den, recip));
    recip = svmul_f16_m(pg, recip, svrecps_f16(den, recip));
    svfloat16_t result = svmul_f16_m(pg, num, recip);
    return result;
}
void MLASCALL
MlasSveErfKernelFp16(const _mlas_fp16_* Input, _mlas_fp16_* Output, size_t N)
{
    // Constants (converted from float32 to float16)
    const __fp16 p = 0.328f;  // Rounded for fp16
    const __fp16 a1 = 0.2505f;
    const __fp16 a2 = -0.2881f;
    const __fp16 a3 = 1.4102f;
    const __fp16 a4 = -1.423f;
    const __fp16 a5 = 1.0547f;
    svfloat16_t vp = svdup_f16(p);
    svfloat16_t va1 = svdup_f16(a1);
    svfloat16_t va2 = svdup_f16(a2);
    svfloat16_t va3 = svdup_f16(a3);
    svfloat16_t va4 = svdup_f16(a4);
    svfloat16_t va5 = svdup_f16(a5);
    svfloat16_t vone = svdup_f16(1.0f);
    svfloat16_t vneg_one = svdup_f16(-1.0f);
    svfloat16_t vzero = svdup_f16(0.0f);
    svfloat16_t vth = svdup_f16(4.0f);  // erf saturates around |x| > 4
    size_t i = 0;
    while (i < N) {
        svbool_t pg = svwhilelt_b16(i, N);
        // Load FP16 input
        svfloat16_t x = svld1_f16(pg, reinterpret_cast<const __fp16*>(&Input[i]));
        // sign mask
        svbool_t neg_mask = svcmplt_f16(pg, x, vzero);
        svfloat16_t sign = svsel_f16(neg_mask, vneg_one, vone);
        svfloat16_t absx = svabs_f16_m(svdup_f16(0), pg, x);
        // use_mask: only compute approximation if abs(x) < 4.0
        svbool_t use_mask = svcmplt_f16(pg, absx, vth);
        // Clamp x for stability
        svfloat16_t absx_clamped = svmin_f16_m(pg, absx, vth);
        // Compute t = 1 / (1 + p*x)
        svfloat16_t denom = svmla_f16_m(pg, vone, vp, absx_clamped);
        svfloat16_t t = svrecpe_f16(denom);
        t = svmul_f16_m(pg, t, svrecps_f16(denom, t));
        t = svmul_f16_m(pg, t, svrecps_f16(denom, t));  // 2 Newton-Raphson
        // Compute polynomial P(t)
        svfloat16_t t2 = svmul_f16_m(pg, t, t);
        svfloat16_t t3 = svmul_f16_m(pg, t2, t);
        svfloat16_t t4 = svmul_f16_m(pg, t3, t);
        svfloat16_t t5 = svmul_f16_m(pg, t4, t);
        svfloat16_t poly = svmul_f16_m(pg, va1, t);
        poly = svmla_f16_m(pg, poly, va2, t2);
        poly = svmla_f16_m(pg, poly, va3, t3);
        poly = svmla_f16_m(pg, poly, va4, t4);
        poly = svmla_f16_m(pg, poly, va5, t5);
        // Compute exp(-x²) using FP16 rational approximation
        svfloat16_t x2 = svmul_f16_m(pg, absx_clamped, absx_clamped);
        svfloat16_t exp_neg_x2 = exp_neg_rational_approx_f16(pg, x2);
        // erf(x) ≈ sign * (1 - P(t) * exp(-x²))
        svfloat16_t poly_mul_exp = svmul_f16_m(pg, poly, exp_neg_x2);
        svfloat16_t one_minus_term = svsub_f16_m(pg, vone, poly_mul_exp);
        svfloat16_t erf_approx = svmul_f16_m(pg, sign, one_minus_term);
        // Clamp to [-1, 1]
        erf_approx = svmin_f16_m(pg, erf_approx, vone);
        erf_approx = svmax_f16_m(pg, erf_approx, vneg_one);
        // Select: use approximation or ±1
        svfloat16_t result = svsel_f16(use_mask, erf_approx, sign);
        // Store FP16 result
        svst1_f16(pg, reinterpret_cast<__fp16*>(&Output[i]), result);
        i += svcntp_b16(svptrue_b16(), pg);
    }
}