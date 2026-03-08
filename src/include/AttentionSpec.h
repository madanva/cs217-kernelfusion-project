#ifndef __ATTENTIONSPEC__
#define __ATTENTIONSPEC__

#include "AxiSpec.h"
#include "GBSpec.h"
#include "NMPSpec.h"

#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>

namespace spec {
  namespace Attention {

    // Reuse NMP fixed-point types for attention computations
    typedef spec::NMP::FixedType FixedType;
    typedef spec::NMP::UnsignedFixedType UnsignedFixedType;
    typedef spec::NMP::AccumType AccumType;
    typedef spec::NMP::UnsignedAccumType UnsignedAccumType;
    typedef spec::NMP::InputFixedType InputFixedType;

    // Vector types for attention
    typedef nvhls::nv_scvector<FixedType, kVectorSize> FixedVectorType;
    typedef nvhls::nv_scvector<AccumType, kVectorSize> AccumVectorType;
    typedef nvhls::nv_scvector<UnsignedFixedType, kVectorSize> UnsignedFixedVectorType;

    // 1/sqrt(d) where d=16: 1/sqrt(16) = 0.25
    const FixedType kInvSqrtD = 0.25f;

    // Maximum sequence length (tile size matches kVectorSize=16)
    const int kMaxSeqLen = 64;
    const int kTileSize = 16;

    class AttentionConfig : public nvhls_message {
      static const int write_width = 128;

    public:
      NVUINT1 is_valid;
      NVUINT3 q_memory_index;
      NVUINT3 k_memory_index;
      NVUINT3 v_memory_index;
      NVUINT3 o_memory_index;
      NVUINT8 seq_len;       // N
      NVUINT8 head_dim;      // d (should be 16)
      NVUINT8 num_tiles;     // ceil(N / kTileSize)

      // Counters
      NVUINT8 row_counter;
      NVUINT8 col_counter;
      NVUINT8 tile_counter;

      // Performance counters
      NVUINT32 cycle_count;
      NVUINT32 gb_read_count;
      NVUINT32 gb_write_count;

      template <unsigned int Size>
      void Marshall(Marshaller<Size>& m) {
        m & is_valid;
        m & q_memory_index;
        m & k_memory_index;
        m & v_memory_index;
        m & o_memory_index;
        m & seq_len;
        m & head_dim;
        m & num_tiles;
        m & row_counter;
        m & col_counter;
        m & tile_counter;
        m & cycle_count;
        m & gb_read_count;
        m & gb_write_count;
      }

      void Reset() {
        is_valid = 0;
        q_memory_index = 0;
        k_memory_index = 1;
        v_memory_index = 2;
        o_memory_index = 3;
        seq_len = 16;
        head_dim = 16;
        num_tiles = 1;
        ResetCounter();
        cycle_count = 0;
        gb_read_count = 0;
        gb_write_count = 0;
      }

      void ResetCounter() {
        row_counter = 0;
        col_counter = 0;
        tile_counter = 0;
      }

      void UpdateRowCounter(bool& is_end) {
        is_end = 0;
        if (row_counter >= (seq_len - 1)) {
          is_end = 1;
          row_counter = 0;
        } else {
          row_counter += 1;
        }
      }

      void UpdateColCounter(bool& is_end) {
        is_end = 0;
        if (col_counter >= (seq_len - 1)) {
          is_end = 1;
          col_counter = 0;
        } else {
          col_counter += 1;
        }
      }

      void UpdateTileCounter(bool& is_end) {
        is_end = 0;
        if (tile_counter >= (num_tiles - 1)) {
          is_end = 1;
          tile_counter = 0;
        } else {
          tile_counter += 1;
        }
      }

      // AXI config write at address region 0xD
      void ConfigWrite(
          const NVUINT16 write_index, const NVUINTW(write_width)& write_data) {
        if (write_index == 0x01) {
          is_valid       = nvhls::get_slc<1>(write_data, 0);
          q_memory_index = nvhls::get_slc<3>(write_data, 8);
          k_memory_index = nvhls::get_slc<3>(write_data, 16);
          v_memory_index = nvhls::get_slc<3>(write_data, 24);
          o_memory_index = nvhls::get_slc<3>(write_data, 32);
          seq_len        = nvhls::get_slc<8>(write_data, 48);
          head_dim       = nvhls::get_slc<8>(write_data, 64);
          num_tiles      = nvhls::get_slc<8>(write_data, 80);
        }
      }

      void ConfigRead(
          const NVUINT16 read_index, NVUINTW(write_width)& read_data) const {
        read_data = 0;
        if (read_index == 0x01) {
          read_data.set_slc<1>(0, is_valid);
          read_data.set_slc<3>(8, q_memory_index);
          read_data.set_slc<3>(16, k_memory_index);
          read_data.set_slc<3>(24, v_memory_index);
          read_data.set_slc<3>(32, o_memory_index);
          read_data.set_slc<8>(48, seq_len);
          read_data.set_slc<8>(64, head_dim);
          read_data.set_slc<8>(80, num_tiles);
        } else if (read_index == 0x02) {
          // Performance counter read
          read_data.set_slc<32>(0, cycle_count);
          read_data.set_slc<32>(32, gb_read_count);
          read_data.set_slc<32>(64, gb_write_count);
        }
      }
    };

  } // namespace Attention
} // namespace spec

#endif
