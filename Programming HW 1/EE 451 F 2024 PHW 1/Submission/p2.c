#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define h 800
#define w 800

#define input_file "input.raw"
#define output_file "output.raw"

int assign_cluster(int data_point, int *mean_array)
{
	int min_distance = 1000, assigned_cluster = -1, i;
	if (!(0 <= data_point <= 255))
	{
		printf("Wrong data ranges");
	}

	for (i = 0; i < 6; i++)
	{
		if (abs(data_point - mean_array[i]) < min_distance)
		{
			min_distance = data_point - mean_array[i];
			assigned_cluster = i;
		}
	}
	if (assigned_cluster == -1)
	{
		perror("Something went wrong with cluster calculation");
	}
	return assigned_cluster;
}

int main(int argc, char **argv)
{
	int i, iters = 30, assignment;
	struct timespec start, stop;
	double time;
	FILE *fp;
	int mean_array[6] = {0, 65, 100, 125, 190, 255};
	int cluster_sum[6], cluster_count[6];

	unsigned char *a = (unsigned char *)malloc(sizeof(unsigned char) * h * w);
	int *assigned_cluster = (int *)malloc(sizeof(int) * h * w);

	for (i = 0; i < h * w; i++)
		assigned_cluster[i] = -1;

	// the matrix is stored in a linear array in row major fashion
	if (!(fp = fopen(input_file, "rb")))
	{
		printf("can not opern file\n");
		return 1;
	}
	fread(a, sizeof(unsigned char), w * h, fp);
	fclose(fp);

	if (clock_gettime(CLOCK_REALTIME, &start) == -1)
		perror("clock gettime");

	for (int iter = 0; iter < iters; iter++)
	{

		// Initializations
		for (i = 0; i < 6; i++)
		{
			cluster_sum[i] = 0;
			cluster_count[i] = 0;
		}

		// Assign clusters
		for (i = 0; i < h * w; i++)
			assigned_cluster[i] = assign_cluster((int)a[i], mean_array);

		// Update Clusters
		for (i = 0; i < h * w; i++)
		{
			cluster_sum[assigned_cluster[i]] += (int)a[i];
			cluster_count[assigned_cluster[i]] += 1;
		}
		for (i = 0; i < 6; i++)
			mean_array[i] = cluster_sum[i] / cluster_count[i];
	}

	// Assign final clusters to the data points.
	for (i = 0; i < h * w; i++)
		a[i] = mean_array[assigned_cluster[i]];

	if (clock_gettime(CLOCK_REALTIME, &stop) == -1)
	{
		perror("clock gettime");
	}
	time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1e9;

	printf("Execution time = %f sec", time);

	if (!(fp = fopen(output_file, "wb")))
	{
		printf("can not opern file\n");
		return 1;
	}
	fwrite(a, sizeof(unsigned char), w * h, fp);
	fclose(fp);

	return 0;
}