/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once
#if defined(ENABLE_FLASH_ATTENTION)

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "core/common/status.h"

namespace onnxruntime {
namespace cuda {
namespace fmha {

struct Qkv_params {
  // The QKV matrices.
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;

  // The stride between rows of the Q, K and V matrices.
  // We're using 32-bit indexing to save registers.
  // The code probably won't work for arrays larger than 2GB.
  uint32_t q_row_stride_in_elts;
  uint32_t k_row_stride_in_elts;
  uint32_t v_row_stride_in_elts;
  uint32_t q_head_stride_in_elts;
  uint32_t k_head_stride_in_elts;
  uint32_t v_head_stride_in_elts;

  // The number of heads.
  int h;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct FMHA_fprop_params : public Qkv_params {
  // The O matrix (output).
  void* __restrict__ o_ptr;

  // The stride between rows of O.
  uint32_t o_row_stride_in_elts;
  uint32_t o_head_stride_in_elts;
  uint32_t o_tmp_row_stride_in_elts;
  uint32_t o_tmp_head_stride_in_elts;

  // The pointer to the O_tmp matrix, which holds O intermediate value during the loop;
  void* __restrict__ o_tmp_ptr;

  // The pointer to the S matrix.
  void* __restrict__ s_ptr;

  // The stride between rows of the S matrix.
  // int64_t s_stride_in_bytes;
  uint32_t s_stride_in_bytes;

  // The pointer to the softmax sum.
  void* __restrict__ softmax_lse_ptr;

  // The dimensions.
  int b, seqlen_q, seqlen_k, d;

  // The scaling factors for the kernel.
  float scale_bmm1f;
  uint32_t scale_bmm1;

  // array of length b+1 holding starting offset of each sequence.
  int* __restrict__ cu_seqlens_q;
  int* __restrict__ cu_seqlens_k;

  bool is_causal;

  int num_splits;  // How many SMs per attention matrix.
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_params>
struct Launch_params {
  Launch_params(const cudaDeviceProp* props_,
                cudaStream_t stream_,
                bool return_softmax_)
      : elts_per_thread(0), props(props_), stream(stream_), return_softmax(return_softmax_) {
  }

  size_t elts_per_thread;

  const cudaDeviceProp* props;

  cudaStream_t stream;

  bool return_softmax;

  Kernel_params params;
  int num_full_heads;
  int num_main_groups;
  int heads_last_wave;
  int main_steps;
  int rest_steps;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
Status run_fmha_fwd_hdim32(Launch_params<FMHA_fprop_params>& launch_params);
Status run_fmha_fwd_hdim64(Launch_params<FMHA_fprop_params>& launch_params);
Status run_fmha_fwd_hdim128(Launch_params<FMHA_fprop_params>& launch_params);

}  // namespace fmha
}  // namespace cuda
}  // namespace onnxruntime

#endif