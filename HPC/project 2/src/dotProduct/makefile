all: dotProduct

dotProduct: dotProduct.cpp walltime.h
	g++-13 -ld_classic -O3 -fopenmp $< -o $@

clean:
	rm -rf dotProduct
