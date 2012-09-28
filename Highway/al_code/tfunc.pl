require my_constants;

# * Have the apprentice choose a speed at time 0 *

print "[1, $middle_lane] ; -1 => [0, $middle_lane, [$left_lane, $red_car_start]] ; [0, $middle_lane, [$middle_lane, $red_car_start]] ; [0, $middle_lane, [$right_lane, $red_car_start]]\n";
print "[1, $middle_lane] ;  0 => [1, $middle_lane, [$left_lane, $red_car_start]] ; [1, $middle_lane, [$middle_lane, $red_car_start]] ; [1, $middle_lane, [$right_lane, $red_car_start]]\n";
print "[1, $middle_lane] ; +1 => [2, $middle_lane, [$left_lane, $red_car_start]] ; [2, $middle_lane, [$middle_lane, $red_car_start]] ; [2, $middle_lane, [$right_lane, $red_car_start]]\n";

foreach $a (@action)
{
foreach $s (@speed)
{
foreach $my_x (@my_car_x)
{
foreach $x (@car_1_x)
{
foreach $y (@car_1_y)
{
	# Choose the new my_x based on the current action, and clip at the boundaries
	$new_my_x = $my_x + ($a * $step);
	$new_my_x = $right_boundary if ($new_my_x > $right_boundary);
	$new_my_x = $left_boundary if ($new_my_x < $left_boundary);

	# Choose the new y based on the current speed, and if boundary is reached, generate a new car
	$new_y = $y + $displace[$s];

	if ($new_y >= $height + $car_length - 10)
	{
		print "[$s, $my_x, [$x, $y]] ; $a => [$s, $new_my_x, [$left_lane, $red_car_start]] ; [$s, $new_my_x, [$middle_lane, $red_car_start]] ; [$s, $new_my_x, [$right_lane, $red_car_start]]\n";
	}
	else
	{
		print "[$s, $my_x, [$x, $y]] ; $a => [$s, $new_my_x, [$x, $new_y]]\n";
	}
}
}
}
}
}
