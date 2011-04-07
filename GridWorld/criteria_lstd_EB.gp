#Plot showing the differences between the different criteria in the ANIRL algorithm
set term epslatex color
set output 'criteria_lstd_EB.eps'
set grid
set xlabel 'Number of samples from the expert'
set key width -100
plot 'criteria_lstd.dat-0' u 1:2 w lines title '$t$' ls 1, 'criteria_lstd.dat-0' u 1:2:2 w errorbars ls 1 notitle,\
'criteria_lstd.dat-1' u 1:2 w lines title '$||\hat\mu^{\pi^{(j)}}(s_0)-\hat\mu^{\pi_E}(s_0)||_2$' ls 2, 'criteria_lstd.dat-1' u 1:2:3 w errorbars ls 2 notitle,\
'criteria_lstd.dat-2' u 1:2 w lines title '$||\mu^{\pi^{(j)}}(s_0)-\mu^{\pi_E}(s_0)||_2$' ls 3, 'criteria_lstd.dat-2' u 1:2:3 w errorbars ls 3 notitle,\
'criteria_lstd.dat-3' u 1:2 w lines title '$||V^{\pi^{(j)}}(s_0)-V^{\pi_E}(s_0)||_2$' ls 4, 'criteria_lstd.dat-3' u 1:2:3 w errorbars ls 4 notitle
