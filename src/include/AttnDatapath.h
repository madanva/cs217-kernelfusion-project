#ifndef __ATTNDATAPATH__
#define __ATTNDATAPATH__

#include <ac_math.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include "Spec.h"
#include "AttentionSpec.h"

// Dot product of two int8 vectors, convert to fixed-point, scale by 1/sqrt(d)
inline void AttnDotProductScale(
    const spec::VectorType& q_vec,
    const spec::VectorType& k_vec,
    spec::Attention::FixedType& result) {

  spec::AccumScalarType dot = 0;
#pragma hls_unroll yes
#pragma cluster addtree
#pragma cluster_type both
  for (int j = 0; j < spec::kVectorSize; j++) {
    dot += q_vec[j] * k_vec[j];
  }

  // Convert accumulator to fixed-point and scale
  spec::Attention::FixedType dot_fixed = dot;
  result = dot_fixed * spec::Attention::kInvSqrtD;
}

// Convert int8 input vector to fixed-point
inline void AttnConvertInputToFixed(
    const spec::VectorType& int_vec,
    spec::Attention::FixedType fixed_vec[spec::kVectorSize]) {

#pragma hls_unroll yes
  for (int i = 0; i < spec::kVectorSize; i++) {
    spec::ScalarType tmp = int_vec[i];
    NVINTW(spec::kIntWordWidth) signed_input = (NVINTW(spec::kIntWordWidth))tmp;
    spec::Attention::InputFixedType in_fixed = 0;
    in_fixed.set_slc(0, signed_input);
    fixed_vec[i] = ConvertFromNmpInputType(in_fixed);
  }
}

// Convert fixed-point output vector to int8
inline void AttnConvertOutputToInt(
    const spec::Attention::FixedType fixed_vec[spec::kVectorSize],
    spec::VectorType& int_vec) {

#pragma hls_unroll yes
  for (int i = 0; i < spec::kVectorSize; i++) {
    spec::NMP::InputFixedType out_tmp = ConvertToNmpOutputType(fixed_vec[i]);
    int_vec[i] = nvhls::get_slc<spec::kIntWordWidth>(out_tmp, 0);
  }
}

// Accumulate: v_accum[k] += exp_val * v_vec[k] (scalar * vector)
inline void AttnWeightedAccum(
    const spec::Attention::UnsignedFixedType& exp_val,
    const spec::Attention::FixedType v_fixed[spec::kVectorSize],
    spec::Attention::AccumType v_accum[spec::kVectorSize]) {

#pragma hls_unroll yes
  for (int k = 0; k < spec::kVectorSize; k++) {
    v_accum[k] += exp_val * v_fixed[k];
  }
}

// Final normalization: output = v_accum / running_sum, convert to int8
inline void AttnFinalNormalize(
    const spec::Attention::AccumType v_accum[spec::kVectorSize],
    const spec::Attention::UnsignedAccumType& running_sum,
    spec::VectorType& output) {

  spec::Attention::AccumType sum_recip;
  ac_math::ac_reciprocal_pwl(running_sum, sum_recip);

#pragma hls_unroll yes
  for (int k = 0; k < spec::kVectorSize; k++) {
    spec::Attention::FixedType normalized = v_accum[k] * sum_recip;
    spec::NMP::InputFixedType out_tmp = ConvertToNmpOutputType(normalized);
    output[k] = nvhls::get_slc<spec::kIntWordWidth>(out_tmp, 0);
  }
}

#endif
