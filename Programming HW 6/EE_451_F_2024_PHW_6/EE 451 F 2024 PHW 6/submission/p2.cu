#include <stdio.h>
#include <cuda.h>
#define N 1024
#define BLOCK_SIZE 16
#define N_STREAMS 4

__global__ void matmul(int *a, int *b, int *c, int offset)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if ((row < N) && (col < N))
    {
        c[(row * N + col) + offset] = 0;
        for (int k = 0; k < N; k++)
        {
            c[(row * N + col) + offset] += a[(row * N + k) + offset] * b[k * N + col];
        }
    }
}

int main()
{
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    int i, j, offset;
    float ms;
    cudaStream_t streams[N_STREAMS];
    cudaEvent_t startEvent, stopEvent;

    for (i = 0; i < N_STREAMS; i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    cudaEventCreate(&startEvent, 0);
    cudaEventCreate(&stopEvent, 0);

    // Allocate host memory
    cudaMallocHost(&h_a, N * N * sizeof(int));
    cudaMallocHost(&h_b, N * N * sizeof(int));
    cudaMallocHost(&h_c, N * N * sizeof(int));

    cudaMalloc(&d_a, N * N * sizeof(int));
    cudaMalloc(&d_b, N * N * sizeof(int));
    cudaMalloc(&d_c, N * N * sizeof(int));

    // Initialize the host arrays
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            h_a[i * N + j] = i;
            h_b[i * N + j] = j;
        }
    }

    dim3 grid(N / (BLOCK_SIZE * N_STREAMS), N / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    cudaEventRecord(startEvent, 0);

    // Synchronously copy b to device
    cudaMemcpy(d_b, h_b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Async
    for (i = 0; i < N_STREAMS; i++)
    {
        offset = i * (N * N / N_STREAMS);
        cudaMemcpyAsync(&d_a[offset], &h_a[offset], N * (N / N_STREAMS) * sizeof(int), cudaMemcpyHostToDevice, streams[i]);
    }

    for (i = 0; i < N_STREAMS; i++)
    {
        offset = i * (N * N / N_STREAMS);
        matmul<<<grid, block, 0, streams[i]>>>(d_a, d_b, d_c, offset);
    }

    for (i = 0; i < N_STREAMS; i++)
    {
        offset = i * (N * N / N_STREAMS);
        cudaMemcpyAsync(&h_c[offset], &d_c[offset], N * (N / N_STREAMS) * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("N Streams: %d\n", N_STREAMS);
    printf("Time taken: %fms\n", ms);
    printf("C[451][451]: %d\n", h_c[451 * N + 451]);

    // Cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    for (i=0; i<N_STREAMS; i++)
        cudaStreamDestroy(streams[i]);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


    return 0;
}