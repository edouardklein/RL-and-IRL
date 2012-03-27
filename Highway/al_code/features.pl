require my_constants;

# The features will be nonnegative and normalized
print "[1, 160] => 0.75 0.5 0.5\n";

foreach $s (@speed)
{
foreach $my_x (@my_car_x)
{
foreach $x (@car_1_x)
{
foreach $y (@car_1_y)
{
	# Blue car coordinates
 	# (exclude border, so that two cars sharing a border does _not_ count as a collision)
	($b_x1, $b_y1, $b_x2, $b_y2) = ($my_x - $car_width + 1, $blue_car_bottom - $car_length + 1, $my_x - 1, $blue_car_bottom - 1);

	# Red car coordinates
	($r_x1, $r_y1, $r_x2, $r_y2) = ($x - $car_width, $y - $car_length, $x, $y);

	# Compute collision
	$eastmost_x1 = (($b_x1 > $r_x1) ? $b_x1 : $r_x1);
	$westmost_x2 = (($b_x2 < $r_x2) ? $b_x2 : $r_x2);
	$southmost_y1 = (($b_y1 > $r_y1) ? $b_y1 : $r_y1);
	$northmost_y2 = (($b_y2 < $r_y2) ? $b_y2 : $r_y2);
	$collide = (($eastmost_x1 > $westmost_x2 || $southmost_y1 > $northmost_y2) ? 0.5 : 0);

	# Compute offroad	
	$offroad = (($my_x < $road_left_boundary || $my_x > $road_right_boundary) ? 0 : 0.5);

	# Compute speed 
	$ns = ($s/$speed[-1] + 1)/2;

	print "[$s, $my_x, [$x, $y]] => $ns $collide $offroad\n";
}
}
}
}
