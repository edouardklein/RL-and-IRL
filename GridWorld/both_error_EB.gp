#Plot comparing the objective performance of mc_ANIRL and lstd_ANIRL
set term epslatex color
set output 'both_error_EB.eps'
set grid
set xlabel 'Number of samples from the expert'
set key width -100
plot 'both_error.dat-0' u 1:2 w lines title 'Monte-Carlo' ls 1, 'both_error.dat-0' u 1:2:3 w errorbars ls 1 notitle,\
'both_error.dat-1' u 1:2 w lines title 'LSTD' ls 2, 'both_error.dat-1' u 1:2:3 w errorbars ls 2 notitle

