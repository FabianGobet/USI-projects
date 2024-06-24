CC=gcc-13
CFLAGS = -g -O3 -fopenmp -ld_classic

all: omp_bug1 omp_bug2 omp_bug3 omp_bug4 omp_bug5

omp_bug1: omp_bug1.c
	$(CC) $(CFLAGS) $< -o $@

omp_bug2: omp_bug2.c
	$(CC) $(CFLAGS) $< -o $@

omp_bug3: omp_bug3.c
	$(CC) $(CFLAGS) $< -o $@

omp_bug4: omp_bug4.c
	$(CC) $(CFLAGS) $< -o $@

omp_bug5: omp_bug5.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf omp_bug1 omp_bug2 omp_bug3 omp_bug4 omp_bug5

