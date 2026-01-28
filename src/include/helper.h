// Copyright 2026 Stanford University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HELPER_H
#define HELPER_H

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "AdpfloatSpec.h"
#include "AdpfloatUtils.h"
#include "SM6Spec.h"
#include "NMPSpec.h"

// =============================================================================
// String Helper Functions
// =============================================================================

/**
 * @brief Split string by delimiter.
 */
std::vector<std::string> split(std::string s, std::string delimiter) {
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  std::string token;
  std::vector<std::string> res;

  while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
    token     = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back(token);
  }

  res.push_back(s.substr(pos_start));
  return res;
}

/**
 * @brief Build N-byte vector from underscore-separated hex string.
 * @param str Hex string like "40_00_10"
 * @return Packed N*8 bit value
 */
template <int num_bytes>
NVUINTW(8 * num_bytes) set_bytes(std::string str) {
  std::string d                     = "_";
  std::vector<std::string> str_list = split(str, d);
  NVUINT8 bytes[num_bytes];

  assert(str_list.size() == num_bytes);
  for (int i = 0; i < num_bytes; i++) {
    std::istringstream converter(str_list[i]);
    unsigned int value;
    converter >> std::hex >> value;
    // cout << hex << value << endl;
    bytes[num_bytes - i - 1] = value;
  }

  NVUINTW(8 * num_bytes) tmp;
  for (int i = 0; i < num_bytes; i++) {
    tmp.set_slc(8 * i, bytes[i]);
  }
  // cout << hex << tmp << endl;
  return tmp;
}

// =============================================================================
// Lab 3 Helper Functions
// =============================================================================

// Tolerance constants for output validation
static const double kAbsTolerance = 0.5;
static const double kPctTolerance = 10.0;

/**
 * @brief Build 128-bit AXI config data for GBCore large buffer.
 * @param num_vec Number of vectors per timestep (bits [7:0])
 * @param base Base address offset in SRAM (bits [31:16])
 * @return Packed 128-bit configuration word
 */
inline NVUINTW(128) make_gbcore_cfg_data(NVUINT8 num_vec, NVUINT16 base) {
  NVUINTW(128) data = 0;
  data.set_slc<8>(0, num_vec);
  data.set_slc<16>(16, base);
  return data;
}

/**
 * @brief Build AXI address for direct GBCore SRAM access.
 * @param local_index SRAM address (bank-interleaved)
 * @return 24-bit AXI address with region selector
 */
inline NVUINTW(24) make_gbcore_data_addr(NVUINT16 local_index) {
  NVUINTW(24) addr = 0;
  addr.set_slc<4>(20, NVUINT4(0x5));
  addr.set_slc<16>(4, local_index);
  return addr;
}

/**
 * @brief Build 128-bit AXI config data for NMP module.
 * @param mode Operation mode: 0 = RMSNorm, 1 = Softmax
 * @param mem Memory index for input/output data
 * @param nvec Number of vectors to process
 * @param ntimestep Number of timesteps to process
 * @param adpbias Adpfloat bias for fixed-point conversion
 * @return Packed 128-bit configuration word
 */
inline NVUINTW(128) make_nmp_cfg_data(
    uint8_t mode,
    uint8_t mem,
    uint8_t nvec,
    uint16_t ntimestep,
    uint8_t adpbias) {
  NVUINTW(128) data = 0;
  data.set_slc<1>(0, NVUINT1(1));
  data.set_slc<3>(8, NVUINT3(mode));
  data.set_slc<3>(32, NVUINT3(mem));
  data.set_slc<8>(48, NVUINT8(nvec));
  data.set_slc<16>(64, NVUINT16(ntimestep));
  data.set_slc<spec::kAdpfloatBiasWidth>(96, spec::AdpfloatBiasType(adpbias));
  return data;
}

/**
 * @brief Convert float vector to adpfloat-encoded VectorType.
 * @param vals Input float values
 * @param bias Adpfloat bias for encoding
 * @return VectorType with adpfloat-encoded elements
 */
inline spec::VectorType make_vector_from_floats(
    const std::vector<float>& vals, spec::AdpfloatBiasType bias) {
  spec::VectorType v = 0;
  for (int i = 0; i < spec::kVectorSize && i < static_cast<int>(vals.size());
       i++) {
    AdpfloatType<spec::kAdpfloatWordWidth, spec::kAdpfloatExpWidth> a;
    spec::NMP::ACFloatType f = vals[i];
    a.set_value_ac_float(f, bias);
    v[i] = a.to_rawbits();
  }
  return v;
}

/**
 * @brief Compute expected RMSNorm output using double precision.
 *
 * RMSNorm formula: y[i] = x[i] / sqrt(mean(x^2) + epsilon)
 * Quantizes inputs to adpfloat first to match hardware behavior.
 *
 * @param vals Input float values
 * @param bias Adpfloat bias for encoding
 * @return Expected output as VectorType (adpfloat encoded)
 */
inline spec::VectorType compute_rms_expected(
    const std::vector<float>& vals, spec::AdpfloatBiasType bias) {
  std::vector<double> vals_full(spec::kVectorSize, 0.0);
  for (int i = 0; i < spec::kVectorSize && i < static_cast<int>(vals.size());
       i++) {
    AdpfloatType<spec::kAdpfloatWordWidth, spec::kAdpfloatExpWidth> a;
    spec::NMP::ACFloatType f = vals[i];
    a.set_value_ac_float(f, bias);
    spec::NMP::ACFloatType qf = a.to_ac_float(bias);
    vals_full[i]              = static_cast<double>(qf.to_double());
  }

  double sum_sq = 0.0;
  for (int i = 0; i < spec::kVectorSize; i++) {
    sum_sq += vals_full[i] * vals_full[i];
  }
  const double inv_size       = 1.0 / static_cast<double>(spec::kVectorSize);
  const double mean           = sum_sq * inv_size;
  const double epsilon        = 1e-4;
  const double rms_reciprocal = 1.0 / std::sqrt(mean + epsilon);

  spec::VectorType out = 0;
  for (int i = 0; i < spec::kVectorSize; i++) {
    const double out_val = vals_full[i] * rms_reciprocal;
    AdpfloatType<spec::kAdpfloatWordWidth, spec::kAdpfloatExpWidth> a;
    a.set_value(static_cast<float>(out_val), bias);
    out[i] = a.to_rawbits();
  }
  return out;
}

/**
 * @brief Compute expected Softmax output using double precision.
 *
 * Softmax formula: y[i] = exp(x[i] - max) / sum(exp(x - max))
 * Uses max subtraction for numerical stability.
 *
 * @param vals Input float values
 * @param bias Adpfloat bias for encoding
 * @return Expected output as VectorType (adpfloat encoded)
 */
inline spec::VectorType compute_softmax_expected(
    const std::vector<float>& vals, spec::AdpfloatBiasType bias) {
  std::vector<double> vals_full(spec::kVectorSize, 0.0);
  for (int i = 0; i < spec::kVectorSize && i < static_cast<int>(vals.size());
       i++) {
    AdpfloatType<spec::kAdpfloatWordWidth, spec::kAdpfloatExpWidth> a;
    spec::NMP::ACFloatType f = vals[i];
    a.set_value_ac_float(f, bias);
    spec::NMP::ACFloatType qf = a.to_ac_float(bias);
    vals_full[i]              = static_cast<double>(qf.to_double());
  }

  double max_val = vals_full[0];
  for (int i = 1; i < spec::kVectorSize; i++) {
    if (vals_full[i] > max_val) {
      max_val = vals_full[i];
    }
  }

  std::vector<double> exp_vals(spec::kVectorSize, 0.0);
  double sum_exp = 0.0;
  for (int i = 0; i < spec::kVectorSize; i++) {
    exp_vals[i] = std::exp(vals_full[i] - max_val);
    sum_exp += exp_vals[i];
  }
  const double inv_sum = (sum_exp == 0.0) ? 0.0 : (1.0 / sum_exp);

  spec::VectorType out = 0;
  for (int i = 0; i < spec::kVectorSize; i++) {
    const double out_val = exp_vals[i] * inv_sum;
    AdpfloatType<spec::kAdpfloatWordWidth, spec::kAdpfloatExpWidth> a;
    a.set_value(static_cast<float>(out_val), bias);
    out[i] = a.to_rawbits();
  }
  return out;
}

/**
 * @brief Convert adpfloat rawbits to double for comparison.
 * @param v Vector containing adpfloat-encoded elements
 * @param idx Element index to convert
 * @param bias Adpfloat bias for decoding
 * @return Double-precision value
 */
inline double adp_to_double(
    const spec::VectorType& v, int idx, spec::AdpfloatBiasType bias) {
  AdpfloatType<spec::kAdpfloatWordWidth, spec::kAdpfloatExpWidth> a(v[idx]);
  float f = a.to_float(bias);
  return static_cast<double>(f);
}

/**
 * @brief Compare output vectors with tolerance-based matching.
 *
 * An element matches if EITHER absolute error <= kAbsTolerance
 * OR percent error <= kPctTolerance.
 *
 * @param actual DUT output vector
 * @param expected Golden reference vector
 * @param bias Adpfloat bias for decoding
 * @return true if all elements match within tolerance
 */
inline bool vectors_match_with_tolerance(
    const spec::VectorType& actual,
    const spec::VectorType& expected,
    spec::AdpfloatBiasType bias) {
  bool ok = true;
  for (int i = 0; i < spec::kVectorSize; i++) {
    const double exp_val = adp_to_double(expected, i, bias);
    const double act_val = adp_to_double(actual, i, bias);
    const double abs_err = std::fabs(act_val - exp_val);
    const double denom   = std::max(std::fabs(exp_val), 1e-9);
    const double pct_err = (abs_err / denom) * 100.0;
    const bool match = !(abs_err > kAbsTolerance | pct_err > kPctTolerance);
    std::cout << (match ? "Match" : "Mismatch") << " idx " << i
              << ": expected=" << exp_val << " actual=" << act_val
              << " abs_err=" << abs_err << " pct_err=" << pct_err << "%"
              << std::endl;
    if (!match) {
      ok = false;
    }
  }
  return ok;
}

#endif
