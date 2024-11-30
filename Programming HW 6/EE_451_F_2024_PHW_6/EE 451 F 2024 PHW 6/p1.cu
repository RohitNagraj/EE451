#include <stdio.h>

#define N 1024
#define BLOCK_SIZE 16
#define N_STREAMS 4

__global__ void matmul(int *a, int *b, int *c, int offset)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if ((row < N) && (col < N))
    {
        c[offset + (row * N + col)] = 0;
        for (int k = 0; k < N; k++)
        {
            c[offset + (row * N + col)] += a[offset + (row * N + k)] * b[k * N + col];
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

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    for (i = 0; i < N_STREAMS; i++)
        cudaStreamCreate(&streams[i]);

    // Allocating host variables
    cudaMallocHost((void **)&h_a, N * N * sizeof(int));
    cudaMallocHost((void **)&h_b, N * N * sizeof(int));
    cudaMallocHost((void **)&h_c, N * N * sizeof(int));

    // Initializing
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            h_a[i * N + j] = i;
            h_b[i * N + j] = j;
            h_c[i * N + j] = 0;
        }
    }

    cudaMalloc(&d_a, N * N * sizeof(int));
    cudaMalloc(&d_b, N * N * sizeof(int));
    cudaMalloc(&d_c, N * N * sizeof(int));

    dim3 grid(N / (BLOCK_SIZE * N_STREAMS), N / BLOCK_SIZE);
    // dim3 grid(16, 64);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    cudaEventRecord(startEvent, 0);
    // Copying B to global memory synchronously
    cudaMemcpy(d_b, h_b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Async
    for (i = 0; i < N_STREAMS; i++)
    {
        offset = i * (N * N / N_STREAMS);
        cudaMemcpyAsync(&d_a[offset], &h_a[offset], (N * N / N_STREAMS) * sizeof(int), cudaMemcpyHostToDevice, streams[i]);
        matmul<<<grid, block, 0, streams[i]>>>(d_a, d_b, d_c, offset);
        cudaMemcpyAsync(&h_c[offset], &d_c[offset], (N * N / N_STREAMS) * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    cudaDeviceSynchronize();

    printf("N Streams: %d\n", N_STREAMS);
    printf("Time taken: %fms\n", ms);
    printf("C[451][451]: %d\n", h_c[451 * N + 451]);

    // Clean Up
    for (i = 0; i < N_STREAMS; i++)
        cudaStreamDestroy(streams[i]);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}