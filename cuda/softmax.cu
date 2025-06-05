/*================================================================
*   Copyright (C) 2024 Fisher. All rights reserved.
*   
*   文件名称：x.cpp
*   创 建 者：YuLianghua
*   创建日期：2024年12月03日
*   描    述：
*
================================================================*/

void softmax_forward_online_cpu(float* out, const float* inp, int N, int C) {
    for (int i=0; i<N; i++) {
        float max_val = INT_MIN;
        double sum = 0.0;
        for (int i=0; i<C; i++) {
            int idx = i*C + j;
            float maxval_prev = max_val;
            if (inp[idx]>max_val) {
                max_val = inp[idx];
                sum = sum*expf(maxval_prev-max_val) + expf(inp[index] - max_val);
            } else {
                sum += expf(inp[index]-max_val);
            }
        }

        float norm = 1.f / (float)sum;
        for (int j=0; j<C; j++) {
            int idx = i*C +j;
            out[idx]=inp[idx]*norm;
        }
    }
}

__global__ void softmax_forward_gpu(float* out, const float* inp, int N, int C) {
    // 每个线程处理一行
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i<N) {
        float max_val = INT_MIN;
        for(int i=0; i<C; i++) {
            int index = i*C + j;
            if (inp[index]>max_val) {
                max_val = inp[index];
            }
        }
        double sum = 0.0;
        for (int j=0; j<C; j++) {
            int index = i*C + j;
            sum += expf(inp[index]-max_val);
        }

        float norm = 1.f / sum;
        for (int j=0; j<C; j++) {
            int index = i*C + j;
            out[index] = expf(inp[index]-max_val) * norm;
        }
    }
}


__global__ void softmax_forward_smem_gpu(float* out, const float* inp, int N, int C) {
    // 一个block 处理一行
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int blocksize = blockDim.x;
    int i = block_size * blockIdx.x;
    const float* x = inp + idx*C;
    float max_val = -INFINITY;
    for (int j=tid; j<C; j+=block_size) {
        max_val = fmaxf(max_val, x[j]);
    }
    shared[tid] = max_val;
    __syncthreads();

    // 规约
    for (int stride=block_size>>1; stride>0; stride>>=1) {
        if (tid<stride) {
            shared[tid] = fmax(shared[tid], share[tid+stride]);
        }
    }
    __syncthreads();
    // max
    float shared_max = shared[0];

    for (int j=tid; j<C; j+=block_size) {
        out[idx*C +j] = expf(x[j]-shared_max);
    }
    __syncthreads();

    x = out + idx*C;
    float sumval = 0.0f;
    for (int j=tid; j<C; j+=block_size) {
        sumval += x[j];
    }
    shared[tid] = sumval;
    __syncthreads();

    // 规约求和
    for (int stride=block_size/2; stride>0; stride>>=1) {
        if (tid<stride) {
            shared[tid] += shared[tid+stride];
        }
    }
    __syncthreads();
    float sum = shared[0];

    for (int j=tid;j<C; j+=block_size) {
        out[idx*C + j] = x[j]/sum;
    }
}

__global__ void softmax_forward_smem_warp_gpu(float* out, const float* inp, int N, int C) {
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32;
    int landId = threadIdx.x % 32;

    int warpPerBlock = blockDim.x / 32;

    float* maxvals = shared;
    float* sumvals = &shared[warpPerBlock];

    const float* x = inp + idx*C;

    float maxval = -INFINITY;
    for (int i=tid; i<C; i+=blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    
    maxval = warp_reduce_max(maxval);  // within-warp reductions for maxval;

    if (laneId==0) {
        maxvals[warpId] = maxval;
    }
    __syncthreads();
    if(tid==0){
        float val = maxvals[tid];
        for (int i=1; i<warpsPerBlock; i++){
            val = fmaxf(val, maxvals[i]);
        }
        maxvals[0] = val;
    }
    __syncthreads();
    float offset = maxvals[0];

    for (int i=tid; i<C; i+=blockDim.x) {
        out[idx*C + i] = expf(x[i]-offset);
    }

    x = out + idx * C;
    float sumval = 0.0f;
    for (int i=tid; i<C; i+=blockDim.x) {
        sum += x[i];
    }
    sumval = warp_reduce_sum(sumval);

    if (laneId==0) {
        sumvals[warpId] = sumval;
    }
    __syncthreads();

    if (tid==0) {
        float val = sumvals[tid];
        for (int i=1; i<warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __synthreads();
    float sum = sumvals[0];

    for (int i=tid; i<C; i+=blockDim.x) {
        out[idx*C + i] = x[i]/sum;
    }
}

__device__ float warp_reduce_max(float val) {
    for (int offset=32>>1; offset>0; offset>>=1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFF, val, offset));
    }
    return val;
}
__device__ float warp_reduce_sum(float val) {
    for (int offset=32>>1; offset>0; offset>>=1) {
        val += __shfl_down_sync(0xFFFFFFF, val, offset);
    }
    return val;
}


__global__ void softmax_forward_online_smem_gpu(float* out, const float* inp, int N, int C){
    const int UNROLL_FACTOR = 8;
    const int warpsPerBlock = blockDim.x / 32; // 一共多少个warp

    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    if (tid >= C) {
        maxvals[warpId] = -INFINITY;
        sumvals[warpId] = 0.0f;
        return;
    }

    const float* x = inp + idx * C; // input
    float* y = out + idx * C; // output

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            maxval = fmaxf(maxval, x[min(C - 1, i + u*blockDim.x)]);
        }
    }

    maxval = warp_reduce_max(maxval);
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();
    if (tid == 0) {
        float val = maxvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        maxvals[0] = val;
    }
    __syncthreads();
    float offset = maxvals[0];
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = __ldcs(&x[min(C - 1, i + u*blockDim.x)]);
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u*blockDim.x < C) {
                float output = expf(reg_array[u] - offset);
                y[min(C - 1, i + u*blockDim.x)] = output; // compiler likes redundant min()?!
                sumval += output; // combined into the same loop unlike kernel3
            }
        }
    }

    sumval = warp_reduce_sum(sumval);
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();
    if (tid == 0) {
        float val = sumvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    float sum = sumvals[0];
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = y[min(C - 1, i + u*blockDim.x)];
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u*blockDim.x < C) {
                y[i + u*blockDim.x] = reg_array[u] / sum;
            }
        }
    }

}


