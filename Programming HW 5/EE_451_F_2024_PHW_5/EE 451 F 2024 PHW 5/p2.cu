#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define N 1024
#define BLOCK_SIZE 32

__global__ void matmul(int *a, int *b, int *c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int res = 0;

    for (int block_id = 0; block_id < N / BLOCK_SIZE; block_id++)
    {
        __shared__ int a_block[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ int b_block[BLOCK_SIZE][BLOCK_SIZE];

        if (row < N && (block_id * BLOCK_SIZE + threadIdx.x) < N)
        {
            a_block[threadIdx.y][threadIdx.x] = a[row * N + (block_id * BLOCK_SIZE + threadIdx.x)];
        }
        else
        {
            a_block[threadIdx.y][threadIdx.x] = 0;
        }

        if (col < N && (block_id * BLOCK_SIZE + threadIdx.y) < N)
        {
            b_block[threadIdx.y][threadIdx.x] = b[(block_id * BLOCK_SIZE + threadIdx.y) * N + col];
        }
        else
        {
            b_block[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++)
            res += a_block[threadIdx.y][k] * b_block[k][threadIdx.x];
        __syncthreads();
    }

    if (row < N && col < N)
    {
        c[row * N + col] = res;
    }
}

int main()
{
    int **a, **b, **c, i, j;
    int *a_d, *b_d, *c_d;
    int *a_h, *b_h, *c_h;

    struct timespec start, stop;
    double time;

    a = (int **)malloc(N * sizeof(int *));
    b = (int **)malloc(N * sizeof(int *));
    c = (int **)malloc(N * sizeof(int *));

    for (i = 0; i < N; i++)
    {
        a[i] = (int *)malloc(sizeof(int) * N);
        b[i] = (int *)malloc(sizeof(int) * N);
        c[i] = (int *)malloc(sizeof(int) * N);
    }

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            a[i][j] = 1;
            b[i][j] = 2;
            c[i][j] = 0;
        }

    // Flattening the 2d array since CUDA expects continuous memory allocation. I know it's redundent,
    // but since the question asks to start with a matrix, I started with a 2D array and now flattening it.
    a_h = (int *)malloc(sizeof(int) * N * N);
    b_h = (int *)malloc(sizeof(int) * N * N);
    c_h = (int *)malloc(sizeof(int) * N * N);

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            a_h[i * N + j] = a[i][j];
            b_h[i * N + j] = b[i][j];
            c_h[i * N + j] = c[i][j];
        }
    }

    cudaMalloc((void **)&a_d, N * N * sizeof(int));
    cudaMalloc((void **)&b_d, N * N * sizeof(int));
    cudaMalloc((void **)&c_d, N * N * sizeof(int));

    cudaMemcpy(a_d, a_h, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(N / BLOCK_SIZE, N / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    if (clock_gettime(CLOCK_REALTIME, &start) == -1)
    {
        perror("clock gettime");
    }
    matmul<<<dimGrid, dimBlock>>>(a_d, b_d, c_d);

    cudaMemcpy(c_h, c_d, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    if (clock_gettime(CLOCK_REALTIME, &stop) == -1)
    {
        perror("clock gettime");
    }
    time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1e9;
    printf("time is %f ns\n", time * 1e9);

    printf("Value of C[451][451] = %d\n", c_h[N * 451 + 451]);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    for (int i = 0; i < N; i++)
    {
        free(a[i]);
        free(b[i]);
        free(c[i]);
    }
    free(a);
    free(b);
    free(c);

    free(a_h);
    free(b_h);
    free(c_h);
}