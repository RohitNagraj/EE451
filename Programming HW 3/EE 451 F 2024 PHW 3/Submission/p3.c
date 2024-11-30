#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define		size	   2*1024*1024

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}


int partition(int *array, int start, int end, int pivot_index) {

    if (pivot_index != -1) swap(&array[end], &array[pivot_index]);
    int pivot = array[end];


    int i = start - 1;
    for (int j = start; j <= end - 1; j++) {
        if (array[j] < pivot) {
            i++;
            swap(&array[i], &array[j]);
        }
    }
    swap(&array[i + 1], &array[end]);
    return i + 1;
}

void quickSort(int *array, int start, int end) {
    if (start < end) {
        int pi = partition(array, start, end, -1);
        quickSort(array, start, pi - 1);
        quickSort(array, pi + 1, end);
    }
}

int main(void) {
    int i, j, tmp;
    struct timespec start, stop;
    double exe_time;
    srand(time(NULL));
    omp_set_num_threads(2);
    int *m = (int *) malloc(sizeof(int) * size);
    for (i = 0; i < size; i++) {
        m[i] = size - i;
        // m[i] = rand();
    }

    if (clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime"); }

    int f = partition(m, 0, size - 1, rand() % size);
    #pragma omp parallel sections
        {
    #pragma omp section
            {
                quickSort(m, 0, f-1);
            }
    #pragma omp section
            {
                quickSort(m, f + 1, size - 1);
            }
        }


    if (clock_gettime(CLOCK_REALTIME, &stop) == -1) { perror("clock gettime"); }
    exe_time = (stop.tv_sec - start.tv_sec) + (double) (stop.tv_nsec - start.tv_nsec) / 1e9;

    for (i = 0; i < 16; i++) printf("%d ", m[i]);
    printf("\nExecution time = %f sec\n", exe_time);
}
