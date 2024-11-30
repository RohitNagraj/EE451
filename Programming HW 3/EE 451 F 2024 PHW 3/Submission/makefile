all: p1a p1b p1c p2a p2b p3

p1a: p1a.c 
	gcc -fopenmp -o p1a p1a.c

p1b: p1b.c
	gcc -fopenmp -o p1b p1b.c

p1c: p1c.c
	gcc -fopenmp -o p1c p1c.c

p2a: p2a.c
	gcc -fopenmp -o p2a p2a.c

p2b: p2b.c
	gcc -fopenmp -o p2b p2b.c

p3: p3.c
	gcc -fopenmp -o p3 p3.c

clean:
	rm p1a p1b p1c p2a p2b p3