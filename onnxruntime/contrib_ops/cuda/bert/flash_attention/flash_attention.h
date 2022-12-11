/******************************************************************************
 * Copyright (c) 2022, Tri Dao.
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

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {
namespace fmha {

size_t get_softmax_lse_size(int max_seqlen_q_, int batch_size, int num_heads);

size_t get_o_tmp_size(int max_seqlen_k_, int total_q, int num_heads, int head_size, int v_head_size);

Status fmha_forward(const cudaDeviceProp& dprops,
                    cudaStream_t stream,
                    void* q,                // shape: (total_q, num_heads, head_size)
                    void* k,                // shape: (total_k, num_heads, head_size)
                    void* v,                // shape: (total_k, num_heads, head_size)
                    void* out,              // shape: (total_q, num_heads, head_size)
                    int32_t* cu_seqlens_q,  // shape: (batch_size + 1)
                    int32_t* cu_seqlens_k,  // shape: (batch_size + 1)
                    void* softmax_lse_buffer,
                    void* o_tmp_buffer,
                    const int batch_size,
                    const int num_heads,
                    const int head_size,
                    const int v_head_size,
                    const int total_q,
                    const int max_seqlen_q_,
                    const int max_seqlen_k_,
                    const float softmax_scale,
                    const bool is_causal,
                    const int num_splits);

}  // namespace fmha
}  // namespace cuda
}  // namespace onnxruntime

#endif