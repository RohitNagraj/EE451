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
    int number, j = 0, sum = 0, total_sum;

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

    MPI_Bcast(array, 64, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        int start = 0, end = 15, i;
        for (i = start; i <= end; i++)
            sum += array[i];
    }
    if (rank == 1)
    {
        int start = 16, end = 31, i;

        for (i = start; i <= end; i++)
            sum += array[i];
    }
    if (rank == 2)
    {
        int start = 32, end = 47, i;

        for (i = start; i <= end; i++)
            sum += array[i];
    }
    if (rank == 3)
    {
        int start = 48, end = 63, i;

        for (i = start; i <= end; i++)
            sum += array[i];
    }

    MPI_Reduce(&sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("\nProcess: %d, Sum: %d\n", rank, total_sum);
    }

    MPI_Finalize();
    return 0;
}