set title "NxN matrix-matrix-multiplication on Apple M2 Pro"
set xlabel "Matrix size (N)"
set ylabel "Performance (GFlop/s)"
set grid
set logscale y 10

set terminal postscript color "Helvetica" 14
set output "timing.ps"

# set terminal png color "Helvetica" 14
# set output "timing.png"

# plot "timing.data" using 2:4 title "square_dgemm" with linespoints


# For performance comparisons

plot "timing_basic_dgemm.data"   using 2:4 title "Naive dgemm" with linespoints, \
     "timing_blocked_dgemm.data" using 2:4 title "Blocked dgemm" with linespoints, \
     "timing_blas_dgemm.data"   using 2:4 title "MKL blas dgemm" with linespoints
