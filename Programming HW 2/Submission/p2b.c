#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#define h 800
#define w 800

#define input_file "input.raw"
#define output_file "output2.raw"

const int NUM_CLUSTERS = 6;
const int NUM_ITERS = 50;
pthread_mutex_t mutex;
pthread_cond_t cv;

typedef struct
{
	int start_index, block_size, thread_id, NUM_THREADS;
	int *threads_done_ptr;
	int *global_mean_array;
	int **thread_cluster_count, **thread_cluster_sum, *assigned_cluster, *global_cluster_sum, *global_cluster_count;
	unsigned char *a;
} thread;

void *assign_cluster(void *thread_data)
{
	thread *data = (thread *)thread_data;
	int i, j, min_distance;
	int iter;

	for (i = 0; i < NUM_CLUSTERS; i++)
	{
		data->global_cluster_count[i] = 0;
		data->global_cluster_sum[i] = 0;
	}

	for (i = 0; i < data->NUM_THREADS; i++)
	{
		for (j = 0; j < NUM_CLUSTERS; j++)
		{
			data->thread_cluster_count[i][j] = 0;
			data->thread_cluster_sum[i][j] = 0;
		}
	}

	for (iter = 0; iter < NUM_ITERS; iter++)
	{

		// Assign clusters
		for (i = data->start_index; i < data->start_index + data->block_size; i++)
		{
			min_distance = 1000;
			for (j = 0; j < NUM_CLUSTERS; j++)
			{
				if (abs((int)data->a[i] - data->global_mean_array[j]) < min_distance)
				{
					min_distance = (int)data->a[i] - data->global_mean_array[j];
					data->assigned_cluster[i] = j;
				}
			}
			if (data->assigned_cluster[i] == -1)
			{
				perror("Something went wrong with cluster calculation");
			}
			data->thread_cluster_count[data->thread_id][data->assigned_cluster[i]] += 1;
			data->thread_cluster_sum[data->thread_id][data->assigned_cluster[i]] += (int)data->a[i];
		}

		pthread_mutex_lock(&mutex);
		if (*(data->threads_done_ptr) < (data->NUM_THREADS - 1))
		{
			*(data->threads_done_ptr) += 1;
			pthread_cond_wait(&cv, &mutex);
		}

		else
		{
			*(data->threads_done_ptr) = 0;
			for (i = 0; i < data->NUM_THREADS; i++)
				for (j = 0; j < NUM_CLUSTERS; j++)
				{
					data->global_cluster_count[j] += data->thread_cluster_count[i][j];
					data->global_cluster_sum[j] += data->thread_cluster_sum[i][j];
				}

			for (i = 0; i < NUM_CLUSTERS; i++)
				data->global_mean_array[i] = data->global_cluster_sum[i] / data->global_cluster_count[i];

			for (i = 0; i < NUM_CLUSTERS; i++)
			{
				data->global_cluster_count[i] = 0;
				data->global_cluster_sum[i] = 0;
			}

			for (i = 0; i < data->NUM_THREADS; i++)
			{
				for (j = 0; j < NUM_CLUSTERS; j++)
				{
					data->thread_cluster_count[i][j] = 0;
					data->thread_cluster_sum[i][j] = 0;
				}
			}
			pthread_cond_broadcast(&cv);
		}
		pthread_mutex_unlock(&mutex);
	}
}

int main(int argc, char **argv)
{

	const int NUM_THREADS = atoi(argv[1]);
	int i, j, iters = 50, assignment, rc, *threads_done_ptr, threads_done = 0;
	struct timespec start, stop;
	double time;
	FILE *fp;
	int global_mean_array[] = {0, 65, 100, 125, 190, 255};
	int global_cluster_sum[NUM_CLUSTERS], global_cluster_count[NUM_CLUSTERS];
	int **thread_cluster_count = (int **)malloc(sizeof(int *) * NUM_THREADS);
	int **thread_cluster_sum = (int **)malloc(sizeof(int *) * NUM_THREADS);
	pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * NUM_THREADS);
	thread *thread_data = (thread *)malloc(sizeof(thread) * NUM_THREADS);
	int block_size = (h * w) / NUM_THREADS;
	pthread_mutex_init(&mutex, NULL);

	unsigned char *a = (unsigned char *)malloc(sizeof(unsigned char) * h * w);
	int *assigned_cluster = (int *)malloc(sizeof(int) * h * w);
	int total_count, total_sum;

	for (i = 0; i < h * w; i++)
		assigned_cluster[i] = -1;

	for (i = 0; i < NUM_THREADS; i++)
	{
		thread_cluster_count[i] = (int *)malloc(sizeof(int) * NUM_CLUSTERS);
		thread_cluster_sum[i] = (int *)malloc(sizeof(int) * NUM_CLUSTERS);
	}

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

	// Initializations
	for (i = 0; i < NUM_CLUSTERS; i++)
	{
		global_cluster_sum[i] = 0;
		global_cluster_count[i] = 0;
	}

	for (i = 0; i < NUM_THREADS; i++)
	{
		for (j = 0; j < NUM_CLUSTERS; j++)
		{
			thread_cluster_count[i][j] = 0;
			thread_cluster_sum[i][j] = 0;
		}
	}

	// Get current assignments.
	for (i = 0; i < NUM_THREADS; i++)
	{
		thread_data[i].thread_id = i;
		thread_data[i].NUM_THREADS = NUM_THREADS;
		thread_data[i].start_index = i * block_size;
		thread_data[i].block_size = block_size;
		thread_data[i].global_mean_array = global_mean_array;
		thread_data[i].a = a;
		thread_data[i].thread_cluster_count = thread_cluster_count;
		thread_data[i].thread_cluster_sum = thread_cluster_sum;
		thread_data[i].assigned_cluster = assigned_cluster;
		thread_data[i].global_cluster_count = global_cluster_count;
		thread_data[i].global_cluster_sum = global_cluster_sum;
		thread_data[i].threads_done_ptr = &threads_done;

		rc = pthread_create(&threads[i], NULL, assign_cluster, (void *)&thread_data[i]);
		if (rc)
		{
			printf("Error: Return code for pthread_create() is %d\n", rc);
			exit(-1);
		}
	}

	// Synchronize threads
	for (i = 0; i < NUM_THREADS; i++)
		pthread_join(threads[i], NULL);

	// Assign final clusters to the data points.
	for (i = 0; i < h * w; i++)
		a[i] = global_mean_array[assigned_cluster[i]];

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