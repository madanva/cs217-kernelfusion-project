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

#ifndef __GBCORESPEC__
#define __GBCORESPEC__

#include <nvhls_int.h>
#include <nvhls_types.h>
#include "AdpfloatSpec.h"
#include "AxiSpec.h"
#include "SM6Spec.h"

namespace spec {
  namespace GB {
    namespace Large {
      // Parameters for Global Buffer
      typedef VectorType WordType;
      const unsigned int kNumWritePorts = 1;  // 1 write port
      const unsigned int kNumReadPorts  = 16; // need at most 16 read ports
      const unsigned int kNumBanks      = 16;
      // total global buffer size = 4096*8banks*16scalars*8bits = 8Mb = 1MB
      const unsigned int kEntriesPerBank = 4096;
      const unsigned int kAddressWidth =
          nvhls::index_width<kNumBanks * kEntriesPerBank>::val;
      const unsigned int kBankIndexSize = nvhls::index_width<kNumBanks>::val;
      const unsigned int kLocalIndexSize =
          nvhls::index_width<kEntriesPerBank>::val;
      typedef NVUINTW(kAddressWidth) Address;
      typedef NVUINTW(kAddressWidth + 1) AddressPlus1;
      typedef NVUINTW(kBankIndexSize) BankIndex;
      typedef NVUINTW(kLocalIndexSize) LocalIndex;
      const int kNumManagers = 4;
    }; // namespace Large


  } // namespace GB
} // namespace spec

// Use of mode
// GBControl  1: Unidirectional, 2: bi-forward, 3: bi-backward, 4: Decoder
// GBLayer    5: MaxPool, 6:MeanPool, 7: LayerAdd
// GBNorm     A: Normalization
// GBAtten    B: Attention
// GBPadding  F: ZeroPadding

class GBCoreConfig {
  static const int write_width       = 128;
  static const int kNumManagersLarge = spec::GB::Large::kNumManagers;
  static const int kAdpBiasWidth     = spec::kAdpfloatBiasWidth;

public:
  // Axi local index 1
  // 0~31
  NVUINT1 is_valid;
  NVUINT1 is_rnn;
  NVUINT1 is_relu;
  NVUINT4 mode;
  // 32~63
  NVUINT3 memory_index[2];
  NVUINT8 num_vector[2];
  // 64~95
  NVUINT16 num_timestep[2];

  // Axi local index 2
  // 0~31   (kNumManagersLarge = 4)
  spec::AdpfloatBiasType adpbias_large[kNumManagersLarge];

  // Axi local index 3
  // 0~63
  NVUINT16 base_large[kNumManagersLarge];


  GBCoreConfig() { Reset(); }

  void Reset() {
    is_valid = 0;
    is_rnn   = 0;
    is_relu  = 0;
    mode     = 0;

#pragma hls_unroll yes
    for (int i = 0; i < 2; i++) {
      memory_index[i] = 0;
      num_vector[i]   = 1;
      num_timestep[i] = 1;
    }

#pragma hls_unroll yes
    for (int i = 0; i < kNumManagersLarge; i++) {
      adpbias_large[i] = 0;
      base_large[i]    = 0;
    }
  }

  void ConfigWrite(
      const NVUINT8 write_index, const NVUINTW(write_width)& write_data) {
    switch (write_index) {
      case 0x1:
        is_valid        = nvhls::get_slc<1>(write_data, 0);
        is_rnn          = nvhls::get_slc<1>(write_data, 8);
        is_relu         = nvhls::get_slc<1>(write_data, 16);
        mode            = nvhls::get_slc<4>(write_data, 24);
        memory_index[0] = nvhls::get_slc<3>(write_data, 32);
        memory_index[1] = nvhls::get_slc<3>(write_data, 40);
        num_vector[0]   = nvhls::get_slc<8>(write_data, 48);
        num_vector[1]   = nvhls::get_slc<8>(write_data, 56);
        num_timestep[0] = nvhls::get_slc<16>(write_data, 64);
        num_timestep[1] = nvhls::get_slc<16>(write_data, 80);
        break;
      case 0x2:
#pragma hls_unroll yes
        for (int i = 0; i < kNumManagersLarge; i++) {
          adpbias_large[i] = nvhls::get_slc<kAdpBiasWidth>(write_data, 8 * i);
        }
        break;
      case 0x3:
#pragma hls_unroll yes
        for (int i = 0; i < kNumManagersLarge; i++) {
          base_large[i] = nvhls::get_slc<16>(write_data, 16 * i);
        }
        break;
      default: break;
    }
  }

  NVUINT16 GetBaseLarge(const NVUINT2 _idx) const { return base_large[_idx]; }

  void ConfigRead(const NVUINT8 read_index, NVUINTW(write_width)& read_data)
      const {
    read_data = 0;
    switch (read_index) {
      case 0x1:
        read_data.set_slc<1>(0, is_valid);
        read_data.set_slc<1>(8, is_rnn);
        read_data.set_slc<1>(16, is_relu);
        read_data.set_slc<4>(24, mode);
        read_data.set_slc<3>(32, memory_index[0]);
        read_data.set_slc<3>(40, memory_index[1]);
        read_data.set_slc<8>(48, num_vector[0]);
        read_data.set_slc<8>(56, num_vector[1]);
        read_data.set_slc<16>(64, num_timestep[0]);
        read_data.set_slc<16>(80, num_timestep[1]);
        break;
      case 0x2:
#pragma hls_unroll yes
        for (int i = 0; i < kNumManagersLarge; i++) {
          read_data.set_slc<kAdpBiasWidth>(8 * i, adpbias_large[i]);
        }
        break;
      case 0x3:
#pragma hls_unroll yes
        for (int i = 0; i < kNumManagersLarge; i++) {
          read_data.set_slc<16>(16 * i, base_large[i]);
        }
        break;
      default: break;
    }
  }
};
#endif
