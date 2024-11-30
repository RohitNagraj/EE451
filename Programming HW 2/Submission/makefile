all: p1a p1b p2a p2b

p1a: p1a.c 
	gcc -lpthread -o p1a p1a.c

p1b: p1b.c
	gcc -lpthread -o p1b p1b.c

p2a: p2a.c
	gcc -lpthread -o p2a p2a.c

p2b: p2b.c
	gcc -lpthread -o p2b p2b.c

clean:
	rm p1a p1b p2a p2b