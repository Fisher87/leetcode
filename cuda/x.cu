#define OFFSET(row, col, ld) ((row)*(ld) + col)

__global__ void sgemm ( float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
                        const int M, const int N, const int K) {
    // 一个BLOCK 计算 BM*BN 个元素，对BM*BN 进行分块，每个线程处理TM*TN, 所以总共有 (BM*BN) / (TM*TN) 个线程；
    const int BM = 128;
    const int BN = 128;
    const int BK=8;
    const int TM=8, TN=8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BM][BK];  // shared memory 存储 tile A 
    __shared__ float s_b[BK][BN];  // shared memory 存储 tile B

    float r_c[TM][TN] = {0.0};    // 使用register，表示的是每个线程计算TM*TN个值;

    // load from global memory
    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by*BM + load_a_smem_m;
    int load_b_gmem_n = bx*BN + load_b_smem_n;

    for (int bk=0; bk<(K+BK-1)/BK; bk++) {
        int load_a_gmem_k = bk*BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
        int load_b_gmem_k = bk*BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET（load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);
        __syncthreads();

#pragma unroll
        for (int k=0; k<BK; k++) {
#pragma unroll
            for (int m=0; m<TM; m++) {
#pragma unroll
                for (int n=0; n<TN; n++) {
                    int comp_a_smem_m = ty*TM + m;
                    int comp_a_smem_n = tx*TN + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_a_smem_n];
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int i=0; i<TM; i++) {
            int store_c_gmem_m = by * BM + ty*TM + i;
#pragma unroll
            for (int j=0; j<TN; j+=4) {
                int store_c_gmem_n = bx*BN + tx*TN + j;
                int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
                FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
            }
        }
    }
}
