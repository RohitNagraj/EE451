#include <stdlib.h>
#include <stdio.h>
#include <time.h>

double **block_matmul(double **A, double **B, double **C, int block_size)
{
	int i, j, k;
	for (i = 0; i < block_size; i++)
		for (j = 0; j < block_size; j++)
			C[i][j] = 0;

	for (i = 0; i < block_size; i++)
		for (j = 0; j < block_size; j++)
			for (k = 0; k < block_size; k++)
				C[i][j] += A[i][k] * B[k][j];

	return C;
}

int main(int argc, char *argv[])
{
	int i, j, k, row, col;
	struct timespec start, stop;
	double time;
	int n = 4096; // matrix size is n*n
	int block_size = atoi(argv[1]);
	int m = n / block_size;

	double **A = (double **)malloc(sizeof(double *) * n);
	double **B = (double **)malloc(sizeof(double *) * n);
	double **C = (double **)malloc(sizeof(double *) * n);

	double **block_A = (double **)malloc(sizeof(double *) * block_size);
	double **block_B = (double **)malloc(sizeof(double *) * block_size);
	double **block_C = (double **)malloc(sizeof(double *) * block_size);

	for (i = 0; i < n; i++)
	{
		A[i] = (double *)malloc(sizeof(double) * n);
		B[i] = (double *)malloc(sizeof(double) * n);
		C[i] = (double *)malloc(sizeof(double) * n);
	}

	for (i = 0; i < block_size; i++)
	{
		block_A[i] = (double *)malloc(sizeof(double) * block_size);
		block_B[i] = (double *)malloc(sizeof(double) * block_size);
		block_C[i] = (double *)malloc(sizeof(double) * block_size);
	}

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			A[i][j] = i;
			B[i][j] = i + j;
			C[i][j] = 0;
		}
	}

	if (clock_gettime(CLOCK_REALTIME, &start) == -1)
	{
		perror("clock gettime");
	}

	for (i = 0; i < m; i++)
		for (k = 0; k < m; k++)
			for (j = 0; j < m; j++)
			{

				// Stoe the blocks in block_A variable.
				for (row = 0; row < block_size; row++)
					for (col = 0; col < block_size; col++)
						block_A[row][col] = A[i * block_size + row][k * block_size + col];

				for (row = 0; row < block_size; row++)
					for (col = 0; col < block_size; col++)
						block_B[row][col] = B[k * block_size + row][j * block_size + col];

				block_matmul(block_A, block_B, block_C, block_size);

				for (row = 0; row < block_size; row++)
					for (col = 0; col < block_size; col++)
						C[i * block_size + row][j * block_size + col] += block_C[row][col];
			}

	if (clock_gettime(CLOCK_REALTIME, &stop) == -1)
	{
		perror("clock gettime");
	}
	time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1e9;

	printf("Number of FLOPs = %lu, Execution time = %f sec,\n%lf MFLOPs per sec\n", (unsigned long)2 * n * n * n, time, 1 / time / 1e6 * 2 * n * n * n);
	printf("C[100][100]=%f\n", C[100][100]);

	// release memory
	for (i = 0; i < block_size; i++)
	{
		free(block_B[i]);
		free(block_A[i]);
		free(block_C[i]);
	}

	for (i = 0; i < n; i++)
	{
		free(A[i]);
		free(B[i]);
		free(C[i]);
	}

	free(block_A);
	free(block_B);
	free(block_C);

	free(A);
	free(B);
	free(C);

	return 0;
}
