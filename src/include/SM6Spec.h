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

#ifndef __SM6SPEC__
#define __SM6SPEC__

#include <TypeToBits.h>
#include <connections/marshaller.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>

namespace spec {
  // Number of PEs
  const int kNumPE = 4;
  // Delay for Trigger signals (start, done)
  const int kGlobalTriggerDelay = 10;

  // Adpfloat definition for floating point representation
  const int kAdpfloatWordWidth = 8;
  const int kAdpfloatExpWidth  = 3; // 0 ~ 7 (or denormal+ 1~7)
  const int kAdpfloatManWidth  = kAdpfloatWordWidth - kAdpfloatExpWidth - 1;
  const int kAdpfloatBiasWidth = 3; // 0 ~ 7
  typedef NVUINTW(kAdpfloatBiasWidth) AdpfloatBiasType;
  const int kAdpfloatOffset = -10;
  // exponent value = Adpfloat.exp + pe.config.adpfloatbias + Offset

  // Vector and Scalar datatype definition
  const int kVectorSize      = 16;
  const int kNumVectorOutput = 1; // cannot be changed anymore
  const int kNumVectorLanes  = kNumVectorOutput * kVectorSize;
  typedef NVUINTW(kAdpfloatWordWidth) ScalarType;
  typedef nvhls::nv_scvector<ScalarType, kVectorSize> VectorType;
  // Half precision type
  typedef NVUINTW(kAdpfloatWordWidth / 2) HalfType;
  typedef nvhls::nv_scvector<HalfType, kVectorSize> HalfVectorType;

  // Accumulation unit reg type
  const int kAccumWordWidth = 32;
  typedef NVINTW(kAccumWordWidth) AccumScalarType;
  typedef typename nvhls::nv_scvector<AccumScalarType, kNumVectorLanes>
      AccumVectorType;

  // Activation unit reg type
  const int kNumActEntries = 4; // Cannot be configured anymore
  const int kActWordWidth  = 20;
  const int kActWordMax    = (1 << (kActWordWidth - 1)) - 1;
  const int kActWordMin    = -kActWordMax;
  const int kActNumFrac    = 14;
  const int kActNumInt     = kActWordWidth - kActNumFrac;
  typedef NVINTW(kActWordWidth) ActScalarType;
  typedef typename nvhls::nv_scvector<ActScalarType, kNumVectorLanes>
      ActVectorType;

  // Attention intermediate storage
  const int kAttentionWordWidth = 32;
  const int kAttentionWordMin   = -(1 << (kAttentionWordWidth - 2));
  const int kAttentionNumFrac   = 20;
  const int kAttentionNumInt    = kAttentionWordWidth - kAttentionNumFrac;
  typedef NVINTW(kAttentionWordWidth) AttentionScalarType;
  // AttentionVectorType has the same width as the VectorType
  typedef typename nvhls::nv_scvector<AttentionScalarType, kNumVectorLanes / 4>
      AttentionVectorType;

} // namespace spec

#endif
