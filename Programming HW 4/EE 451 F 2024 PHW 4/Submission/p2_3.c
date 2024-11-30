#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int array[64], partial_array[16], partial_sums[4];
    int number, j = 0;

    if (rank == 0)
    {

        FILE *file = fopen("number.txt", "r");
        if (file == NULL)
        {
            printf("Error: Could not open file.\n");
            return 1;
        }
        while (fscanf(file, "%d", &number) != EOF)
        {
            array[j] = number;
            j++;
        }
    }

    MPI_Scatter(array, 16, MPI_INT, partial_array, 16, MPI_INT, 0, MPI_COMM_WORLD);


    int start = 0, end = 15, i, sum=0;
    for (i = start; i <= end; i++)
        sum += partial_array[i];

    MPI_Gather(&sum, 1, MPI_INT, partial_sums, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        int total_sum = partial_sums[0] + partial_sums[1] + partial_sums[2] + partial_sums[3];
        printf("\nProcess: %d, Sum: %d\n", rank, total_sum);
    }

    MPI_Finalize();
    return 0;
}