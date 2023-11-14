CC=mpicc
CFLAGS=-O3 -I/opt/homebrew/include/libpng16 -L/opt/homebrew/lib -lpng

all: mandel_mpi

mandel_mpi:  mandel_mpi.c pngwriter.c
	$(CC) $^ $(CFLAGS) -o $@

clean:
	rm -f mandel_mpi

