set term postscript enhanced color
set output "courbe_a.ps"
set grid
set xlabel "Iterations"
set ylabel "||{/Symbol m}_E - {/Symbol m}_{bar}||_2"
plot "courbe_a.dat" u 1:2 w lines smooth csplines ls 1 notitle

