CC = gcc
CFLAGS = -O3 -march=native -std=c99 -pedantic -Wall -Wextra -Wshadow -Wpointer-arith -Wcast-qual -Wstrict-prototypes -Wmissing-prototypes

all:
	$(CC) fht.c -c $(CFLAGS)
	$(CC) test_float.c fht.o -o test_float $(CFLAGS)
	$(CC) test_double.c fht.o -o test_double $(CFLAGS)

clean:
	rm -f fht.o test_float test_double
