set term postscript enhanced color
set output "courbe_a.ps"
set grid
set xlabel "Iterations"
set ylabel "||{/Symbol m}_E - {/Symbol m}_{bar}||_2"
plot "courbe_a.dat" u 2:4 smooth csplines notitle, "courbe_a.dat" u 2:4 w points ls 1 notitle

