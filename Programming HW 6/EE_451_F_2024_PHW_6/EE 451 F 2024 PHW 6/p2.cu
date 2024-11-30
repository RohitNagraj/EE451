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
    int *h_a, *h_a2, *h_b, *h_c;
    int *d_a, *d_a2, *d_b, *d_c;
    int i, j;
    float ms;
    cudaStream_t streams[N_STREAMS];
    cudaEvent_t startEvent, stopEvent;

    for (i = 0; i < N_STREAMS; i++)
        cudaStreamCreate(&streams[i]);

    cudaEventCreate(&startEvent, 0);
    cudaEventCreate(&stopEvent, 0);

    // Allocate host memory
    cudaMallocHost(&h_a, N * N * sizeof(int));
    cudaMallocHost(&h_a2, N * N * sizeof(int));
    cudaMallocHost(&h_b, N * N * sizeof(int));
    cudaMallocHost(&h_c, N * N * sizeof(int));

    cudaMalloc(&d_a, N * N * sizeof(int));
    cudaMalloc(&d_a2, N * N * sizeof(int));
    cudaMalloc(&d_b, N * N * sizeof(int));
    cudaMalloc(&d_c, N * N * sizeof(int));

    // Initialize the host arrays
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            h_a[i * N + j] = i;
            h_a2[i * N + j] = i;
            h_b[i * N + j] = j;
        }
    }

    dim3 grid(N / (BLOCK_SIZE * N_STREAMS), N / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    cudaEventRecord(startEvent, 0);

    // Synchronously copy b to device
    cudaMemcpy(d_b, h_b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Async

    cudaMemcpyAsync(&d_a[0 * (N * N / N_STREAMS)], &h_a[0 * (N * N / N_STREAMS)], N * (N / N_STREAMS) * sizeof(int), cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(&d_a[1 * (N * N / N_STREAMS)], &h_a[1 * (N * N / N_STREAMS)], N * (N / N_STREAMS) * sizeof(int), cudaMemcpyHostToDevice, streams[1]);
    cudaMemcpyAsync(&d_a[2 * (N * N / N_STREAMS)], &h_a[2 * (N * N / N_STREAMS)], N * (N / N_STREAMS) * sizeof(int), cudaMemcpyHostToDevice, streams[2]);
    cudaMemcpyAsync(&d_a[3 * (N * N / N_STREAMS)], &h_a[3 * (N * N / N_STREAMS)], N * (N / N_STREAMS) * sizeof(int), cudaMemcpyHostToDevice, streams[3]);

    matmul<<<grid, block, 0, streams[0]>>>(d_a, d_b, d_c, 0 * (N * N / N_STREAMS));
    matmul<<<grid, block, 0, streams[1]>>>(d_a, d_b, d_c, 1 * (N * N / N_STREAMS));
    matmul<<<grid, block, 0, streams[2]>>>(d_a, d_b, d_c, 2 * (N * N / N_STREAMS));
    matmul<<<grid, block, 0, streams[3]>>>(d_a, d_b, d_c, 3 * (N * N / N_STREAMS));

    cudaMemcpyAsync(&h_c[0 * (N * N / N_STREAMS)], &d_c[0 * (N * N / N_STREAMS)], N * (N / N_STREAMS) * sizeof(int), cudaMemcpyDeviceToHost, streams[3]);
    cudaMemcpyAsync(&h_c[1 * (N * N / N_STREAMS)], &d_c[1 * (N * N / N_STREAMS)], N * (N / N_STREAMS) * sizeof(int), cudaMemcpyDeviceToHost, streams[2]);
    cudaMemcpyAsync(&h_c[2 * (N * N / N_STREAMS)], &d_c[2 * (N * N / N_STREAMS)], N * (N / N_STREAMS) * sizeof(int), cudaMemcpyDeviceToHost, streams[1]);
    cudaMemcpyAsync(&h_c[3 * (N * N / N_STREAMS)], &d_c[3 * (N * N / N_STREAMS)], N * (N / N_STREAMS) * sizeof(int), cudaMemcpyDeviceToHost, streams[0]);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    printf("N Streams: %d\n", N_STREAMS);
    printf("Time taken: %fms\n", ms);
    printf("C[451][451]: %d\n", h_c[451 * N + 451]);

    // Cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    for (i = 0; i < N_STREAMS; i++)
        cudaStreamDestroy(streams[i]);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}