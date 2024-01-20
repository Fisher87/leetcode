/*
   虽然在v1中使用shared memory 可以有效提升HBM访存效率，但是会存在bank conflict问题；
   什么是bank conflict 问题？
   在同一个线程块（thread block）中的线程共享一块 Shared memory。Shared memory 被分割为 32 个逻辑块（banks），
   不同的逻辑块可以被多个线程同时访问。连续的 32-bit 访存被分配到连续的逻辑块（bank）
   在cuda 的 share memory 中是通过 bank 进行管理的，一般来说有32个bank
     bank |  0   1   2  ...   31 
    ----------------------------------
    warp0 |  0   1   2  ...   31
    warp1 | 32  33  34  ...   63
    ...


   不同的线程同时访问同一个bank的不同
   地址就会出现bank conflict 问题；
*/


#define OFFSET(row, col, ld) ((row)*(ld) + col)

__global__ void sgemm_v1(
 float * __restrict__ a, float * __restrict__ b, float* __restrict__ c,
 const int M, const int N, const int K) {
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;   // x+y*D_x

    __shared__ float s_a[BK][BM];
    __shared__ float s_b[BK][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;        //row of s_a 
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;        //col of s_b
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk=0; bk<(K+BK-1)/BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        // 转化为使用列存储方式;
        s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();

        #pragma unroll
        for (int tk=0; tk<BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty*TM/2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty*TM/2 + BM/2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx*TN/2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx*TN/2 + BN/2]);

#pragma unroll
            for (int tm=0; tm<TM; tm++) {
#pragma unroll
                for (int tn=0; tn<TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for(int i=0; i<TM/2; i++) {
        int store_c_gmem_m = by*BM + ty*TM/2 + i ;
        int store_c_gmem_n = bx*BN + tx*TN/2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN/2]) = FLOAT4(r_c[i][4]);
    }

#pragma unroll
    for (int i=0; i<TM/2; i++) {
        int store_c_gmem_m = by*BM + BM/2 + ty*TM/2 + i;
        int store_c_gmem_n = bx*BN + tx * TN/2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM/2][0]);
        FLOAT4(c[store_c_gmem_addr + BN/2]) = FLOAT4(r_c[i + TM/2][4]);
    }
}
