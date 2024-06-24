CC=gcc-13

all: recur_seq recur_omp

recur_seq: recur_seq.c
	$(CC) -O3 -ld_classic $^ -o $@

recur_omp: recur_omp.c
	$(CC) -O3 -fopenmp -ld_classic $^ -o $@ -lm

clean:
	rm -rf recur_seq recur_omp

