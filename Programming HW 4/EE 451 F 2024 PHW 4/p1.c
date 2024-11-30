#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        int msg, source, destination, sending_tag, receiving_tag;

        msg = 451;
        source = 3;
        sending_tag = 1;
        receiving_tag = 0;
        destination = 1;

        MPI_Send(&msg, 1, MPI_INT, destination, sending_tag, MPI_COMM_WORLD);
        printf("Process %d: Initially Msg = %d\n", rank, msg);
        MPI_Recv(&msg, 1, MPI_INT, source, receiving_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d: Received Msg = %d. Done!\n", rank, msg);
    }
    if (rank == 1)
    {
        int msg, source, destination, sending_tag, receiving_tag;
        source = 0;
        sending_tag = 2;
        receiving_tag = 1;
        destination = 2;
        MPI_Recv(&msg, 1, MPI_INT, source, receiving_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        msg++;
        printf("Process %d: Msg = %d\n", rank, msg);
        MPI_Send(&msg, 1, MPI_INT, destination, sending_tag, MPI_COMM_WORLD);
    }
    if (rank == 2)
    {
        int msg, source, destination, sending_tag, receiving_tag;
        source = 1;
        sending_tag = 3;
        receiving_tag = 2;
        destination = 3;
        MPI_Recv(&msg, 1, MPI_INT, source, receiving_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        msg++;
        printf("Process %d: Msg = %d\n", rank, msg);
        MPI_Send(&msg, 1, MPI_INT, destination, sending_tag, MPI_COMM_WORLD);
    }
    if (rank == 3)
    {
        int msg, source, destination, sending_tag, receiving_tag;
        source = 2;
        sending_tag = 0;
        receiving_tag = 3;
        destination = 0;
        MPI_Recv(&msg, 1, MPI_INT, source, receiving_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        msg++;
        printf("Process %d: Msg = %d\n", rank, msg);
        MPI_Send(&msg, 1, MPI_INT, destination, sending_tag, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}