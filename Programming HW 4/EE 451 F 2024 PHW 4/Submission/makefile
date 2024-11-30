all: p1 p2_1 p2_2 p2_3

p1: p1.c 
	mpicc -o p1 p1.c

p2_1: p2_1.c
	mpicc -o p2_1 p2_1.c

p2_2: p2_2.c
	mpicc -o p2_2 p2_2.c

p2_3: p2_3.c
	mpicc -o p2_3 p2_3.c

clean:
	rm p1 p2_1 p2_2 p2_3