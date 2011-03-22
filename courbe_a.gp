set term postscript enhanced color
set output "courbe_a.ps"
set grid
set xlabel "Iterations"
set ylabel "||{/Symbol m}_E - {/Symbol m}_{bar}||_2"
plot "courbe_a.dat" u 2:4 smooth csplines title "Critere d'arret" ls 1, "courbe_a.dat" u 2:4 w points ls 1 notitle,\
"courbe_a.dat" u 2:5 smooth csplines title "Erreur mesuree" ls 2, "courbe_a.dat" u 2:5 w points ls 2 notitle,\
"courbe_a.dat" u 2:6 smooth csplines title "Erreur vraie" ls 3, "courbe_a.dat" u 2:6 w points ls 3 notitle,\
"courbe_a.dat" u 2:7 smooth csplines title "Diff performance" ls 4, "courbe_a.dat" u 2:7 w points ls 4 notitle
