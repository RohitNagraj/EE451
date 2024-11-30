#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int array[64];
    int number, j = 0;

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

    if (rank == 0)
    {
        int start = 0, end = 15, i, sum = 0, partial_sum_1, partial_sum_2, partial_sum_3;

        for (i = start; i <= end; i++)
            sum += array[i];

        MPI_Recv(&partial_sum_1, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&partial_sum_2, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&partial_sum_3, 1, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        sum += partial_sum_1 + partial_sum_2 + partial_sum_3;
        printf("\nProcess: %d, Sum: %d\n", rank, sum);
    }
    if (rank == 1)
    {
        int start = 16, end = 31, i, sum = 0, destination = 0;

        for (i = start; i <= end; i++)
            sum += array[i];
        MPI_Send(&sum, 1, MPI_INT, destination, 0, MPI_COMM_WORLD);
    }
    if (rank == 2)
    {
        int start = 32, end = 47, i, sum = 0, destination = 0;

        for (i = start; i <= end; i++)
            sum += array[i];
        MPI_Send(&sum, 1, MPI_INT, destination, 0, MPI_COMM_WORLD);
    }
    if (rank == 3)
    {
        int start = 48, end = 63, i, sum = 0, destination = 0;

        for (i = start; i <= end; i++)
            sum += array[i];
        MPI_Send(&sum, 1, MPI_INT, destination, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}