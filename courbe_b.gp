set term postscript enhanced color
set output "courbe_b.ps"
set grid
set xlabel "Number of samples"
set ylabel "||{/Symbol m}_E - {/Symbol m}_{bar}||_2"
set log x
set yrange [-0.5:8]
plot "courbe_b_50.dat" u 3:4 smooth csplines ls 2 title "M=5", "courbe_b_50.dat" u 3:4 w points ls 2 notitle,\
"courbe_b_100.dat" u 3:4 smooth csplines ls 3 title "M=10", "courbe_b_100.dat" u 3:4 w points ls 3 notitle,\
"courbe_b_150.dat" u 3:4 smooth csplines ls 4 title "M=15", "courbe_b_150.dat" u 3:4 w points ls 4 notitle,\
"courbe_b_Nous.dat" u 3:4 title "LSTD{/Symbol m}" ls 1
