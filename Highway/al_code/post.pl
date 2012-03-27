open FEATURES, "perl features.pl |";
open POLICY, "policy.dat";
open POLICY_TEXT, ">policy.txt";

# Add state labels to each line
while(<FEATURES>)
{
        @s = split /=>/, $_;
	$_ = <POLICY>;
        print POLICY_TEXT &trim($s[0]), ";", $_;
}

close FEATURES;
close POLICY;
close POLICY_TEXT;

# Subroutines

sub trim($)
{
        my $string = shift;
        $string =~ s/^\s+//;
        $string =~ s/\s+$//;
        return $string;
}

