// coil_field_cuda.cu
// ==================
// CUDA kernels for magnetic field computation (circular coil + Biot-Savart).
//
// Compiled when CYNA_CUDA_ENABLED is defined (i.e. nvcc is present and
// xmake is invoked with --with-cuda=y / setup.py detects nvcc).
//
// Host-callable entry points (extern "C"):
//   cyna_coil_circular_field_cuda  — analytic ring-coil field on GPU
//   cyna_coil_biot_savart_cuda     — Biot-Savart for filamentary coil on GPU
//
// Both functions allocate GPU buffers, launch the kernel, copy results to the
// caller-supplied CPU output arrays, and free GPU buffers.  They are called
// from flt_bindings.cpp when CUDA is available.

#ifdef CYNA_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>

// ---------------------------------------------------------------------------
// Device-side elliptic integrals (float32 AGM, 8/10 iterations)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float _kf(float m) {
    float a = 1.0f, b = sqrtf(fmaxf(1.0f - m, 0.0f));
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float an = 0.5f * (a + b);
        b = sqrtf(a * b);
        a = an;
    }
    return 1.5707963267948966f / a;
}

__device__ __forceinline__ float _ef(float m) {
    float a = 1.0f, b = sqrtf(fmaxf(1.0f - m, 0.0f));
    float S = 0.5f * m, power = 1.0f;
#pragma unroll
    for (int i = 0; i < 10; ++i) {
        float c  = 0.5f * (a - b);
        float an = 0.5f * (a + b);
        b = sqrtf(a * b);
        a = an;
        S += power * c * c;
        power *= 2.0f;
    }
    return 1.5707963267948966f / a * (1.0f - S);
}

// ---------------------------------------------------------------------------
// Device: ring-coil analytic Brho, Bz
// ---------------------------------------------------------------------------

__device__ __forceinline__ void _coil_BrhoBz(
    float a, float rho, float z, float I,
    float& Brho, float& Bz_)
{
    const float MU0_2PI = 2.0e-7f;
    if (rho < 1e-9f) {
        float r = sqrtf(a * a + z * z);
        Brho = 0.0f;
        Bz_  = MU0_2PI * I * a * a / (r * r * r);
        return;
    }
    float Q  = (a + rho) * (a + rho) + z * z;
    float sQ = sqrtf(Q);
    float k2 = 4.0f * a * rho / Q;
    if (k2 >= 1.0f) k2 = 1.0f - 1.2e-7f;
    float K  = _kf(k2), E_ = _ef(k2);
    float dE = (a - rho) * (a - rho) + z * z;
    if (dE < 1e-20f) dE = 1e-20f;
    float c  = MU0_2PI * I;
    Brho = c * z / (rho * sQ) * (-K + E_ * (a*a + rho*rho + z*z) / dE);
    Bz_  = c / sQ             * ( K + E_ * (a*a - rho*rho - z*z) / dE);
}

// ---------------------------------------------------------------------------
// Kernel: circular coil field at all field points
// ---------------------------------------------------------------------------
__global__ void _circular_coil_kernel(
    float cx, float cy, float cz,
    float nx, float ny, float nz,
    float a,  float current,
    const float* __restrict__ xyz,   // (N, 3) Cartesian field points
    int N,
    float* __restrict__ BR,
    float* __restrict__ BPhi,
    float* __restrict__ BZ)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float px = xyz[3*i], py = xyz[3*i+1], pz = xyz[3*i+2];
    float dx = px-cx, dy = py-cy, dz = pz-cz;
    float z_l = nx*dx + ny*dy + nz*dz;
    float rx = dx - z_l*nx, ry = dy - z_l*ny, rz = dz - z_l*nz;
    float rho = sqrtf(rx*rx + ry*ry + rz*rz);

    float Brho_l, Bz_l;
    _coil_BrhoBz(a, rho, z_l, current, Brho_l, Bz_l);

    float bx, by, bz_lab;
    if (rho > 1e-9f) {
        float inv = 1.0f / rho;
        bx    = Brho_l * rx * inv + Bz_l * nx;
        by    = Brho_l * ry * inv + Bz_l * ny;
        bz_lab= Brho_l * rz * inv + Bz_l * nz;
    } else {
        bx = Bz_l * nx; by = Bz_l * ny; bz_lab = Bz_l * nz;
    }

    float phi = atan2f(py, px);
    float cp_ = cosf(phi), sp_ = sinf(phi);
    BR[i]   =  bx * cp_ + by * sp_;
    BPhi[i] = -bx * sp_ + by * cp_;
    BZ[i]   =  bz_lab;
}

// ---------------------------------------------------------------------------
// Kernel: Biot-Savart for one arbitrary filamentary coil
// ---------------------------------------------------------------------------
__global__ void _biot_savart_kernel(
    const float* __restrict__ seg_starts,  // (N_seg, 3)
    const float* __restrict__ seg_ends,    // (N_seg, 3)
    int   N_seg,
    float current,
    const float* __restrict__ xyz,         // (N_pts, 3)
    int   N_pts,
    float* __restrict__ BR,
    float* __restrict__ BPhi,
    float* __restrict__ BZ)
{
    const float MU0_4PI = 1.0e-7f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_pts) return;

    float px = xyz[3*i], py = xyz[3*i+1], pz = xyz[3*i+2];
    float bx = 0.0f, by = 0.0f, bz = 0.0f;

    for (int s = 0; s < N_seg; ++s) {
        float x0 = seg_starts[3*s], y0 = seg_starts[3*s+1], z0 = seg_starts[3*s+2];
        float x1 = seg_ends[3*s],   y1 = seg_ends[3*s+1],   z1 = seg_ends[3*s+2];
        float dlx = x1-x0, dly = y1-y0, dlz = z1-z0;
        float r0x = px-x0, r0y = py-y0, r0z = pz-z0;
        float r1x = px-x1, r1y = py-y1, r1z = pz-z1;
        float R0 = sqrtf(r0x*r0x + r0y*r0y + r0z*r0z);
        float R1 = sqrtf(r1x*r1x + r1y*r1y + r1z*r1z);
        if (R0 < 5e-4f || R1 < 5e-4f) continue;
        float dot  = r0x*r1x + r0y*r1y + r0z*r1z;
        float dnom = R0 * R1 * (R0*R1 + dot);
        if (fabsf(dnom) < 1e-30f) continue;
        float fac = MU0_4PI * current * (R0 + R1) / dnom;
        bx += fac * (dly*r0z - dlz*r0y);
        by += fac * (dlz*r0x - dlx*r0z);
        bz += fac * (dlx*r0y - dly*r0x);
    }

    float phi = atan2f(py, px);
    float cp_ = cosf(phi), sp_ = sinf(phi);
    BR[i]   =  bx * cp_ + by * sp_;
    BPhi[i] = -bx * sp_ + by * cp_;
    BZ[i]   =  bz;
}

// ---------------------------------------------------------------------------
// Host entry points (called from flt_bindings.cpp)
// ---------------------------------------------------------------------------

#define THREADS_PER_BLOCK 256
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        return false; \
    } \
} while(0)

/// Compute circular coil field on GPU.
/// xyz_cpu : (N,3) float32 host array
/// BR/BPhi/BZ_cpu : (N,) float32 host output arrays (pre-allocated)
extern "C"
bool cyna_coil_circular_field_cuda(
    float cx, float cy, float cz,
    float nx, float ny, float nz,
    float a,  float current,
    const float* xyz_cpu, int N,
    float* BR_cpu, float* BPhi_cpu, float* BZ_cpu)
{
    size_t sz3 = (size_t)N * 3 * sizeof(float);
    size_t sz1 = (size_t)N     * sizeof(float);

    float *d_xyz = nullptr, *d_BR = nullptr, *d_BP = nullptr, *d_BZ = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xyz, sz3));
    CUDA_CHECK(cudaMalloc(&d_BR,  sz1));
    CUDA_CHECK(cudaMalloc(&d_BP,  sz1));
    CUDA_CHECK(cudaMalloc(&d_BZ,  sz1));

    CUDA_CHECK(cudaMemcpy(d_xyz, xyz_cpu, sz3, cudaMemcpyHostToDevice));

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    _circular_coil_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        cx, cy, cz, nx, ny, nz, a, current, d_xyz, N, d_BR, d_BP, d_BZ);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(BR_cpu,   d_BR, sz1, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(BPhi_cpu, d_BP, sz1, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(BZ_cpu,   d_BZ, sz1, cudaMemcpyDeviceToHost));

    cudaFree(d_xyz); cudaFree(d_BR); cudaFree(d_BP); cudaFree(d_BZ);
    return true;
}

/// Compute Biot-Savart field for one filamentary coil on GPU.
/// seg_starts_cpu, seg_ends_cpu : (N_seg,3) float32 host arrays
/// xyz_cpu : (N,3) float32 host array
/// BR/BPhi/BZ_cpu : (N,) float32 host output arrays (pre-allocated)
extern "C"
bool cyna_coil_biot_savart_cuda(
    const float* seg_starts_cpu, const float* seg_ends_cpu,
    int N_seg, float current,
    const float* xyz_cpu, int N,
    float* BR_cpu, float* BPhi_cpu, float* BZ_cpu)
{
    size_t szS  = (size_t)N_seg * 3 * sizeof(float);
    size_t sz3  = (size_t)N     * 3 * sizeof(float);
    size_t sz1  = (size_t)N         * sizeof(float);

    float *d_ss = nullptr, *d_se = nullptr, *d_xyz = nullptr;
    float *d_BR = nullptr, *d_BP = nullptr, *d_BZ = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ss,  szS));
    CUDA_CHECK(cudaMalloc(&d_se,  szS));
    CUDA_CHECK(cudaMalloc(&d_xyz, sz3));
    CUDA_CHECK(cudaMalloc(&d_BR,  sz1));
    CUDA_CHECK(cudaMalloc(&d_BP,  sz1));
    CUDA_CHECK(cudaMalloc(&d_BZ,  sz1));

    CUDA_CHECK(cudaMemcpy(d_ss,  seg_starts_cpu, szS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_se,  seg_ends_cpu,   szS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xyz, xyz_cpu,         sz3, cudaMemcpyHostToDevice));

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    _biot_savart_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_ss, d_se, N_seg, current, d_xyz, N, d_BR, d_BP, d_BZ);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(BR_cpu,   d_BR, sz1, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(BPhi_cpu, d_BP, sz1, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(BZ_cpu,   d_BZ, sz1, cudaMemcpyDeviceToHost));

    cudaFree(d_ss); cudaFree(d_se); cudaFree(d_xyz);
    cudaFree(d_BR); cudaFree(d_BP); cudaFree(d_BZ);
    return true;
}

#endif // CYNA_CUDA_ENABLED
