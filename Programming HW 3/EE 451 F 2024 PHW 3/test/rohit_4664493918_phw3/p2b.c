#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include<omp.h>

#define		num_of_points	   40000000

typedef struct {
    double x;
    double y;
} Point;

int main(void) {
    int i, j;
    int num_of_points_in_circle = 0;
    double pi;
    struct timespec start, stop;
    double time;

    omp_set_num_threads(2);


    Point *data_point = (Point *) malloc(sizeof(Point) * num_of_points);
    for (i = 0; i < num_of_points; i++) {
        data_point[i].x = (double) rand() / (double) RAND_MAX;
        data_point[i].y = (double) rand() / (double) RAND_MAX;
    }

    if (clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime"); }

#pragma omp parallel sections private(i) reduction(+:num_of_points_in_circle)
    {
#pragma omp section
        {
            for (i = 0; i < num_of_points / 2; i++) {
                if ((data_point[i].x - 0.5) * (data_point[i].x - 0.5) + (data_point[i].y - 0.5) * (
                        data_point[i].y - 0.5) <=
                    0.25) {
                    num_of_points_in_circle += 1;
                }
            }
        }
#pragma omp section
        {
            for (i = num_of_points / 2; i < num_of_points; i++) {
                if ((data_point[i].x - 0.5) * (data_point[i].x - 0.5) + (data_point[i].y - 0.5) * (
                        data_point[i].y - 0.5) <=
                    0.25) {
                    num_of_points_in_circle += 1;
                }
            }
        }
    }

    if (clock_gettime(CLOCK_REALTIME, &stop) == -1) { perror("clock gettime"); }
    time = (stop.tv_sec - start.tv_sec) + (double) (stop.tv_nsec - start.tv_nsec) / 1e9;

    pi = 4 * (double) num_of_points_in_circle / (double) num_of_points;
    printf("Estimated pi is %f, execution time = %f sec\n", pi, time);
    free(data_point);
}
