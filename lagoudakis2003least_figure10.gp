set term postscript enhanced color
set output "lagoudakis2003least_figure10.ps"
set grid
set xrange[0.8:4.2]
set xlabel "States"
set ylabel "Q"
set xtics 1
plot "lagoudakis2003least_figure10.dat" u 1:2 w lines smooth csplines ls 1 title "Q(s,R)", "lagoudakis2003least_figure10.dat" u 1:2:3 w errorbars ls 1 lw 1 notitle,"lagoudakis2003least_figure10.dat" u 1:4 w lines smooth csplines ls 3 title "Q(s,L)", "lagoudakis2003least_figure10.dat" u 1:4:5 w errorbars ls 3 lw 2 notitle
