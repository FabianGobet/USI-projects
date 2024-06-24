all: hist_seq hist_omp

hist_seq: hist_seq.cpp
	g++-13 -O3 -ld_classic $^ -o $@

hist_omp: hist_omp.cpp
	g++-13 -O3 -ld_classic -fopenmp $^ -o $@

clean:
	rm -rf hist_seq hist_omp

