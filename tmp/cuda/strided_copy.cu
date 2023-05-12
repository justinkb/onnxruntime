#include <iostream>
#include <fstream>
#include <string>

#define Status cudaError_t
#define CUDA_CALL(expr)                                                 \
  do {                                                                  \
    cudaError_t err = (expr);                                           \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA Error %s:%d\n", __FILE__, __LINE__);        \
      fprintf(stderr, "CUDA Error Code  : %d\n     Error String: %s\n", \
              err, cudaGetErrorString(err));                            \
      exit(err);                                                        \
    }                                                                   \
  } while (0)

template <typename T>
__global__ void StridedCopy(const T* in, const int H, longlong4 in_stride,  // coord (b,n,s,h)
                            T* out, longlong4 out_stride                    // coord (b,n,s,h)
) {
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;
  if (h < H) {
    const int in_offset = b * in_stride.x + n * in_stride.y + s * in_stride.z + h * in_stride.w;
    const int out_offset = b * out_stride.x + n * out_stride.y + s * out_stride.z + h * out_stride.w;
    out[out_offset] = in[in_offset];
  }
}

template <typename T>
__global__ void StridedCopyLarge(const T* in, const int H, longlong4 in_stride,  // coord (b,n,s,h)
                                 T* out, longlong4 out_stride                    // coord (b,n,s,h)
) {
  // Use when (H*)*num_heads > 1024
  int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;

  const int h_step = blockDim.x;

  while (h < H) {
    const int in_offset = b * in_stride.x + n * in_stride.y + s * in_stride.z + h * in_stride.w;
    const int out_offset = b * out_stride.x + n * out_stride.y + s * out_stride.z + h * out_stride.w;
    out[out_offset] = in[in_offset];
    h += h_step;
  }
}

template <int NumBytes>
struct ToByteType;

template <>
struct ToByteType<2> {
  using T = uchar2;
};

template <>
struct ToByteType<4> {
  using T = uint;
};

template <>
struct ToByteType<8> {
  using T = uint2;
};

template <>
struct ToByteType<16> {
  using T = uint4;
};

template <>
struct ToByteType<32> {
  using T = ulonglong4;
};

template <int NumBytes>
using ToBytes = typename ToByteType<NumBytes>::T;

template <typename T>
Status LaunchStridedCopy(hipStream_t stream,
                         const T* in, int4 in_shape, longlong4 in_stride,  // coord (b,n,s,h)
                         T* out, longlong4 out_stride,                     // coord (b,n,s,h)
                         int max_threads_per_block) {
  int batch_size = in_shape.x;
  int num_heads = in_shape.y;
  int sequence_length = in_shape.z;
  int head_size = in_shape.w;

  const dim3 grid(sequence_length, batch_size);
  if (0 == (head_size % 4)) {
    using Bytes = ToBytes<sizeof(T) * 4>;
    const int H = head_size / 4;
    in_stride.x /= 4;
    in_stride.y /= 4;
    in_stride.z /= 4;
    out_stride.x /= 4;
    out_stride.y /= 4;
    out_stride.z /= 4;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      StridedCopy<Bytes><<<grid, block, 0, stream>>>(reinterpret_cast<const Bytes*>(in), H, in_stride,
                                                     reinterpret_cast<Bytes*>(out), out_stride);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      StridedCopyLarge<Bytes><<<grid, block, 0, stream>>>(reinterpret_cast<const Bytes*>(in), H, in_stride,
                                                          reinterpret_cast<Bytes*>(out), out_stride);
    }
  } else if (0 == (head_size % 2)) {
    using Bytes = ToBytes<sizeof(T) * 2>;
    const int H = head_size / 2;
    in_stride.x /= 2;
    in_stride.y /= 2;
    in_stride.z /= 2;
    out_stride.x /= 2;
    out_stride.y /= 2;
    out_stride.z /= 2;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      StridedCopy<Bytes><<<grid, block, 0, stream>>>(reinterpret_cast<const Bytes*>(in), H, in_stride,
                                                     reinterpret_cast<Bytes*>(out), out_stride);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      StridedCopyLarge<Bytes><<<grid, block, 0, stream>>>(reinterpret_cast<const Bytes*>(in), H, in_stride,
                                                          reinterpret_cast<Bytes*>(out), out_stride);
    }
  } else {
    using Bytes = ToBytes<sizeof(T)>;
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      StridedCopy<Bytes><<<grid, block, 0, stream>>>(reinterpret_cast<const Bytes*>(in), head_size, in_stride,
                                                     reinterpret_cast<Bytes*>(out), out_stride);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      StridedCopyLarge<Bytes><<<grid, block, 0, stream>>>(reinterpret_cast<const Bytes*>(in), head_size, in_stride,
                                                          reinterpret_cast<Bytes*>(out), out_stride);
    }
  }
  return cudaGetLastError();
}

template <typename T1, typename T2>
int64_t inner(T1 a, T2 b) {
  return int64_t(a.x) * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

template <typename T>
__global__ void fill_linear(T* in, int64_t num_elem) {
  int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < num_elem) {
    in[i] = 0.0001 * (i + 1);
  }
}

template <typename T>
void TestStridedCopy(const std::string& name,
                     int4 in_pyhsical_shape, int4 in_logical_shape,  // coord (b,n,s,h)
                     int4 out_physical_shape, int4 out_offset,       // coord (b,n,s,h)
                     int max_threads_per_block = 1024) {
  T* in;
  T* out;
  int64_t in_num_elem = in_pyhsical_shape.x * in_pyhsical_shape.y * in_pyhsical_shape.z * in_pyhsical_shape.w;
  int64_t out_num_elem = out_physical_shape.x * out_physical_shape.y * out_physical_shape.z * out_physical_shape.w;
  std::cout << __FILE__ ":" << __LINE__ << " in_num_elem:" << in_num_elem << " out_num_elem:" << out_num_elem << std::endl;
  CUDA_CALL(cudaMalloc(&in, sizeof(T) * in_num_elem + 1024));
  CUDA_CALL(cudaMalloc(&out, sizeof(T) * out_num_elem + 1024));

  longlong4 in_stride{in_pyhsical_shape.y * in_pyhsical_shape.z * in_pyhsical_shape.w, in_pyhsical_shape.z * in_pyhsical_shape.w, in_pyhsical_shape.w, 1};
  longlong4 out_stride{out_physical_shape.y * out_physical_shape.z * out_physical_shape.w, out_physical_shape.z * out_physical_shape.w, out_physical_shape.w, 1};

  fill_linear<<<(in_num_elem - 1) / 256 + 1, 256>>>(in, in_num_elem);
  CUDA_CALL(cudaDeviceSynchronize());
  CUDA_CALL(cudaMemset(out, 0, sizeof(T) * out_num_elem));
  CUDA_CALL(LaunchStridedCopy(0,
                              in, in_logical_shape, in_stride,
                              out + inner(out_offset, out_stride), out_stride, max_threads_per_block));
  CUDA_CALL(cudaDeviceSynchronize());

  std::vector<char> buffer;
  {
    std::cout << __FILE__ ":" << __LINE__ << " " << in_num_elem * sizeof(T) << std::endl;
    buffer.resize(in_num_elem * sizeof(T));
    CUDA_CALL(cudaMemcpy(buffer.data(), in, buffer.size(), cudaMemcpyDeviceToHost));
    std::ofstream f(name + "_in.bin", std::ios::binary);
    f.write(buffer.data(), buffer.size());
  }

  {
    std::cout << __FILE__ ":" << __LINE__ << " " << out_num_elem * sizeof(T) << std::endl;
    buffer.resize(out_num_elem * sizeof(T));
    CUDA_CALL(cudaMemcpy(buffer.data(), out, buffer.size(), cudaMemcpyDeviceToHost));
    std::ofstream f(name + "_out.bin", std::ios::binary);
    f.write(buffer.data(), buffer.size());
  }
}

int main(int argc, char* argv[]) {
  int4 in_pyhsical_shape{4, 4, 4, 16};
  int4 in_logical_shape{4, 4, 1, 16};
  int4 out_physical_shape{4, 4, 3, 16};
  int4 out_offset{0, 0, 1, 0};
  if (argc > 1) {
    in_pyhsical_shape = int4{std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4])};
    in_logical_shape = int4{std::atoi(argv[5]), std::atoi(argv[6]), std::atoi(argv[7]), std::atoi(argv[8])};
    out_physical_shape = int4{std::atoi(argv[9]), std::atoi(argv[10]), std::atoi(argv[11]), std::atoi(argv[12])};
    out_offset = int4{std::atoi(argv[13]), std::atoi(argv[14]), std::atoi(argv[15]), std::atoi(argv[16])};
  }

  CUDA_CALL(cudaSetDevice(15));
  TestStridedCopy<float>("basic", in_pyhsical_shape, in_logical_shape, out_physical_shape, out_offset);
}
