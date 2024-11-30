cmake_minimum_required(VERSION 3.29)
project(EE_451_F_2024_PHW_4 C)

set(CMAKE_C_STANDARD 11)

add_executable(EE_451_F_2024_PHW_4
    "mpi examples/all_to_one_reduction.c"
    "mpi examples/bcast.c"
    "mpi examples/create_communicator.c"
    "mpi examples/gather.c"
    "mpi examples/helloworld.c"
    "mpi examples/scatter.c")
