#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>

typedef struct
{
    int row, col, block_size, matrix_size;
    double **A, **B, **C;
} thread_2d;

pthread_mutex_t mutex;

void *MatMul(void *thread_data)
{
    int i, j, k;
    double sum;
    thread_2d *data = ((thread_2d *)thread_data);

    for (i = data->row; i < data->row + data->block_size; i++)
        for (j = 0; j < data->matrix_size; j++)
        {
            sum = 0;
            for (k = data->col; k < data->col + data->block_size; k++)
                sum += data->A[i][k] * data->B[k][j];
            pthread_mutex_lock(&mutex);
            data->C[i][j] += sum;
            pthread_mutex_unlock(&mutex);
        }
    // printf("Row: %d, Col: %d, i: %d, j: %d, k: %d, block_size: %d\n", data->row, data->col, i, j, k, data->block_size);
}

int main(int argc, char *argv[])
{
    const int NUM_THREADS_ROOT = atoi(argv[1]);
    pthread_t **threads;
    thread_2d **thread_data;
    int rc, i, j;
    int n = 4096;
    int block_size = n / NUM_THREADS_ROOT;
    struct timespec start, stop;
    double time;
    pthread_mutex_init(&mutex, NULL);

    // Initialize matrices
    double **A = (double **)malloc(sizeof(double *) * n);
    double **B = (double **)malloc(sizeof(double *) * n);
    double **C = (double **)malloc(sizeof(double *) * n);
    for (i = 0; i < n; i++)
    {
        A[i] = (double *)malloc(sizeof(double) * n);
        B[i] = (double *)malloc(sizeof(double) * n);
        C[i] = (double *)malloc(sizeof(double) * n);
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

    threads = (pthread_t **)malloc(NUM_THREADS_ROOT * sizeof(pthread_t *));
    thread_data = (thread_2d **)malloc(NUM_THREADS_ROOT * sizeof(thread_2d *));

    // Allocate memory for threads
    for (i = 0; i < NUM_THREADS_ROOT; i++)
    {
        threads[i] = (pthread_t *)malloc(NUM_THREADS_ROOT * sizeof(pthread_t));
        thread_data[i] = (thread_2d *)malloc(NUM_THREADS_ROOT * sizeof(thread_2d));
    }

    if (clock_gettime(CLOCK_REALTIME, &start) == -1)
    {
        perror("clock gettime");
    }

    for (i = 0; i < NUM_THREADS_ROOT; i++)
    {
        for (j = 0; j < NUM_THREADS_ROOT; j++)
        {
            thread_data[i][j].row = i * block_size;
            thread_data[i][j].col = j * block_size;
            thread_data[i][j].block_size = block_size;
            thread_data[i][j].matrix_size = n;
            thread_data[i][j].A = A;
            thread_data[i][j].B = B;
            thread_data[i][j].C = C;

            rc = pthread_create(&threads[i][j], NULL, MatMul, (void *)&thread_data[i][j]);
            if (rc)
            {
                printf("Error: Return code for pthread_create() is %d\n", rc);
                exit(-1);
            }
        }
    }

    // Wait for all threads to complete
    for (i = 0; i < NUM_THREADS_ROOT; i++)
    {
        for (j = 0; j < NUM_THREADS_ROOT; j++)
        {
            pthread_join(threads[i][j], NULL);
        }
    }

    if (clock_gettime(CLOCK_REALTIME, &stop) == -1)
    {
        perror("clock gettime");
    }
    time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1e9;

    printf("Number of FLOPs = %lu, Execution time = %f sec,\n%lf MFLOPs per sec\n", (unsigned long)2 * n * n * n, time, 1 / time / 1e6 * 2 * n * n * n);
    printf("C[100][100]=%f\n", C[100][100]);

    for (i = 0; i < n; i++)
    {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }

    for (i = 0; i < NUM_THREADS_ROOT; i++)
    {
        free(threads[i]);
        free(thread_data[i]);
    }

    free(A);
    free(B);
    free(C);

    free(threads);
    free(thread_data);
}