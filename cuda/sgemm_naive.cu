#define OFFSET(row, col, ld) ((row)*(ld) + col)

__global__ void naiveSgemm(
                           float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
                           const int M, const int N, const int K) {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    int m = blockDim.y * blockIdx.y + threadIdx.y;
    if (m<M and n<N) {
        float psum = 0.0;
        #pragma unroll
        for(int k=0; k<K; k++) {
            psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m,n,N)] = psum;
    }
}

const int BM=32, BN=32;
const int M=512, N = 512, K=512;
// 使用总共 M*N 个线程进行处理，每个block 线程块中使用 BM*BN个
// 由于最终的结果形式为 M*N，而在cuda中线程分布是不同的, 一维坐标对应到cuda中为y值，二维坐标对应的才是x值
// 所以对应的cuda显存编排方式为(BN, BM)
dim3 blockDim(BN, BM);
dim3 gridDim((N+BN-1)/BN, (M+BM-1)/BM);


