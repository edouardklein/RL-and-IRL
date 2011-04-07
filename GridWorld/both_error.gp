#Plot comparing the objective performance of mc_ANIRL and lstd_ANIRL
set term epslatex color
set output 'both_error.eps'
set grid
set xlabel 'Number of samples from the expert'
set key width -100
plot 'both_error.dat' u 1:2 smooth csplines title 'LSTD' ls 1, 'both_error.dat' u 1:2 w points ls 1 notitle,\
'both_error.dat' u 3:4 smooth csplines title 'Monte-Carlo' ls 2, 'both_error.dat' u 3:4 w points ls 2 notitle
