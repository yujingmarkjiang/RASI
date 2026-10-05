#include <cuda_runtime.h>

extern "C" int rasi_cuda_device_count() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
  return n;
}

__global__ void gru_gates(const float* x, const float* h, const float* Wz, const float* Wr,
                          const float* Uz, const float* Ur, const float* bz, const float* br,
                          float* z, float* r, int I, int H) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= H) return;
  float az = bz[i], ar = br[i];
  for (int j = 0; j < I; ++j) {
    az += Wz[i * I + j] * x[j];
    ar += Wr[i * I + j] * x[j];
  }
  for (int j = 0; j < H; ++j) {
    az += Uz[i * H + j] * h[j];
    ar += Ur[i * H + j] * h[j];
  }
  z[i] = 1.f / (1.f + expf(-az));
  r[i] = 1.f / (1.f + expf(-ar));
}

__global__ void gru_candidate(const float* x, const float* h, const float* r, const float* z,
                              const float* Wh, const float* Uh, const float* bh, float* hnew,
                              int I, int H) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= H) return;
  float ah = bh[i];
  for (int j = 0; j < I; ++j) ah += Wh[i * I + j] * x[j];
  for (int j = 0; j < H; ++j) ah += Uh[i * H + j] * (r[j] * h[j]);
  float hh = tanhf(ah);
  hnew[i] = (1.f - z[i]) * h[i] + z[i] * hh;
}
