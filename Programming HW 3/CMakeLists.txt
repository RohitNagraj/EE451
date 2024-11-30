cmake_minimum_required(VERSION 3.29)
project(Programming_HW_3 C)

set(CMAKE_C_STANDARD 11)

find_package(OpenMP REQUIRED)

add_executable(Programming_HW_3
        #        "EE 451 F 2024 PHW 3/p1a.c"
        #        "EE 451 F 2024 PHW 3/p1b.c"
#        "EE 451 F 2024 PHW 3/p1c.c"
        #        "EE 451 F 2024 PHW 3/p2_serial.c"
#                "EE 451 F 2024 PHW 3/p2a.c"
#                "EE 451 F 2024 PHW 3/p2b.c"
#                "EE 451 F 2024 PHW 3/p3a.c"
#                "EE 451 F 2024 PHW 3/p3b.c"
                "EE 451 F 2024 PHW 3/p3c.c"
)

# Link OpenMP to the target
if (OpenMP_C_FOUND)
    target_link_libraries(Programming_HW_3 OpenMP::OpenMP_C)
endif ()