# Make features array (while also making state-to-num hash)
$_ = `perl features.pl | wc`;
$num_states = (split)[0];
open MAKE_F, ">make_F.m";
print MAKE_F "function F = make_F\n";
print MAKE_F "F = zeros($num_states, 3);\n"; 	# TODO: Make this less hard-code-y
open FEATURES, "perl features.pl |";
while(<FEATURES>)
{
	($state, $features) = split /=>/, $_;
	$n++;
	$hash{&trim($state)} = $n;
	@features = split /\s+/, &trim($features);
	foreach $i (1 .. @features)
	{
		print MAKE_F "F($n, $i) = $features[$i-1];\n";
	}
}
close FEATURES;
close MAKE_F;

# Make transition array
open MAKE_THETA, ">make_THETA.m";
print MAKE_THETA "function THETA = make_THETA\n";
print MAKE_THETA "THETA = zeros($num_states, 3, $num_states);\n"; 	# TODO: Make this less hard-code-y
open TFUNC, "perl tfunc.pl |";
while(<TFUNC>)
{
        ($state_action, $states) = split /=>/, $_;
        ($state, $action) = split /;/, $state_action;
	$state = $hash{&trim($state)};
        $action = remap_action(&trim($action)); 	# This is done so that "none" action breaks ties
        @states = split /;/, $states;
        $prob = 1/@states;
        foreach $s (@states)
        {
		$s = $hash{&trim($s)};
                print MAKE_THETA "THETA($state, $action, $s) = $prob;\n";
        }
}
print MAKE_THETA "THETA = sparse(reshape(permute(THETA, [2 1 3]), $num_states*3, $num_states));\n"; 	# TODO: Make this less hard-code-y
close TFUNC;
close MAKE_THETA;

# Subroutines
	
sub trim($)
{
	my $string = shift;
	$string =~ s/^\s+//;
	$string =~ s/\s+$//;
	return $string;
}

sub remap_action($)
{
	my $num = shift;
	return 1 if ($num == 0);
	return 2 if ($num == -1);
	return 3 if ($num == 1);
}

