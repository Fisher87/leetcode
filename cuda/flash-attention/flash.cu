
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


// grid_dim = (batch_size, num_heads)
// block_dim = (Bc)       一个block 处理一个batch 的一个head (N * d)
__global__
void forward_kernel(const float* Q, const float* K, const float* V, 
                    const int N, const int d, const int Tc, const int Tr, const int Br, 
                    const int Bc, const float* scale, float* l, float* m, float* O) {
    int tid = threadIdx.x;
    int bx  = blockIdx.x; // batch index
    int by  = blockIdx.y; // head index

    int qkv_offset = (bx * gridDim.y + by) * N * d; // gridDim.y = num_heads
    int lm_offset = (bx * gridDim.y + by) * N;

    // Define sram for Q, K, V, S
    extern __shared__ float sram[];
    int tile_size = Bc * d; // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[2 * tile_size];
    float* S = &sram[3 * tile_size];

    for (int j=0; j<Tc; j++){
        // load Kj, Vj to sram; each size is Bc * d ==> each thread load d elements
        for (int x=0; x<d; x++ ) {
            Kj[(tid*d)+x] = K[qkv_offset + (tile_size * j) + (tid*d)+x ];
            Vj[(tid*d)+x] = V[qkv_offset + (tile_size * j) + (tid*d)+x ];
        }
        __syncthreads();

        for (int i=0; i< Tr; i++) {
            // load Qi to sram, l and m to registers
            for (int x=0; x<d; x++) {
                Qi[(tid*d)+x] = Q[qkv_offset + (tile_size * j) + (tid*d)+x ];
            }
            float row_m_pre = m[lm_offset + (Br * i) + tid];
            float row_l_pre = l[lm_offset + (Br * i) + tid];

            // compute S=QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y=0; y<Bc; y++) {
                float sum = 0;
                for (int x =0; x<d; x++) {
                    sum += Qi[(tid * d) + x] * Kj[(y * d) + x];
                }
                sum *= scale;
                S[(tid * Bc) + y] = sum;
                if (sum > row_m) {
                    row_m = sum;
                }
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y=0; y<Bc; y++ ){
                S[(tid * Bc) + y] = __expf(S[(tid * Bc) + y] - row_m);
                row_l += S[(tid * Bc) + y];
            }

            // compute new l and m;
            float row_m_new = max(row_m_pre, row_m);
            // l_new = l_pre * exp(m_pre - m_new) + l * exp(m - m_new)
            float row_l_new = (__expf(row_m_pre - row_m_new)*row_l_pre + __expf(row_m - row_m_new)*row_l);

            // write O, l, m to HBM
            for (int x=0; x<d; x++){
                float pv = 0;  // Pij * Vj
                for (int y=0; y<Bc; y++) {
                    pv += S[(tid * Bc) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size*i) + (tid*d) + x] = (1/row_l_new) \
                * ((row_l_pre * __expf(row_m_pre-row_m_new) * O[qkv_offset + (tile_size*j) + (tid*d) + x]) \
                + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tid] = row_m_new;
            l[lm_offset + (Br * i) + tid] = row_l_new;
        }
        __syncthreads();
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int Bc = 32;
    const int Br = 32;
    const int B = Q.size(0);
    const int nH = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    const int Tc = ceil((float)N / Bc);
    const int Tr = ceil((float)N / Br);
    const float scale = 1.0 / sqrt(d);

    // initialize O, l, m to HBM;
    auto O = torch::zeros_like(Q)
    auto l = torch::zeros({B, nH, N})
    auto m = torch::zeros({B, nH, N}, -INFINITY)
    torch::Device device(torch::kCUDA);
    l = l.to(device);
    m = m.to(device);

    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Bc * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(B, nH);
    dim3 block_dim(Bc);
    forward_kernel<<<grid_dim, block_dim, sram_size>>>(Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), 
             N, d, Tc, Tr, Br, Bc, scale, 
             l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>());
    return O;
}

/* main.cpp
#include <torch/extension.h>

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
}
*/

/* python test code 

import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Load the CUDA kernel as a python module
minimal_attn = load(name='minimal_attn', sources=['main.cpp', 'flash.cu'], extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 16
n_head = 12
seq_len = 64
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

print('=== profiling manual attention ===')

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_attn(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling minimal flash attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result = minimal_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('attn values sanity check:', torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-02))
*/