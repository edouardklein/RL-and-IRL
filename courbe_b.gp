set term postscript enhanced color
set output "courbe_b.ps"
set grid
set xlabel "Nb Samples"
set ylabel "||{/Symbol m}_E - {/Symbol m}_{bar}||_2"
plot "courbe_b.dat" u 3:4 smooth csplines title "Critere d'arret" ls 1, "courbe_b.dat" u 3:4 w points ls 1 notitle,\
"courbe_b.dat" u 3:5 w lines title "Erreur mesuree" ls 2, "courbe_b.dat" u 3:5 w points ls 2 notitle,\
"courbe_b.dat" u 3:6 w lines title "Erreur vraie" ls 3, "courbe_b.dat" u 3:6 w points ls 3 notitle,\
"courbe_b.dat" u 3:7 w lines title "Diff performance" ls 4, "courbe_b.dat" u 3:7 w points ls 4 notitle
