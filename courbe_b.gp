set term postscript enhanced color
set output "courbe_b.ps"
set grid
set xlabel "Number of samples"
set ylabel "||{/Symbol m}_E - {/Symbol m}_{bar}||_2"
plot "courbe_b_50.dat" u 3:4 smooth csplines ls 1 title "M=5", "courbe_b_50.dat" u 3:4 w points ls 1 notitle,\
"courbe_b_100.dat" u 3:4 smooth csplines ls 2 title "M=10", "courbe_b_100.dat" u 3:4 w points ls 2 notitle,\
"courbe_b_150.dat" u 3:4 smooth csplines ls 3 title "M=15", "courbe_b_150.dat" u 3:4 w points ls 3 notitle,\
"courbe_b_200.dat" u 3:4 smooth csplines ls 4 title "M=20", "courbe_b_200.dat" u 3:4 w points ls 4 notitle

