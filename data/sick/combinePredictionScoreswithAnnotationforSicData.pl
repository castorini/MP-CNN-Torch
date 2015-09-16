use strict;
use warnings;


#20      There is no man in a black jacket doing tricks on a motorbike   A person in a black jacket is doing tricks on a motorbike       3.6     CONTRADICTION

my $anno = shift;
my $pred = shift;

open IN1, "< $anno" or die $!;
open IN2, "< $pred" or die $!;

my @annoo = <IN1>;
my @predd = <IN2>;

chomp(@annoo);
chomp(@predd);

if(@annoo-1 != @predd){
	print("error!\n");
	exit;
}

print "pair_ID\tsentence_A\tsentence_B\trelatedness_score\tentailment_judgment\n";

for(my $i = 0; $i < @predd; $i++){
	if($annoo[$i+1] =~ /^(\d+\t(.*\t){2})(\d+\.*\d*)(\t.*)$/){
		print $1.$predd[$i].$4."\n";
	}else{
		print "Noooooooooooooooooooooot possible\n";
		print $annoo[$i+1]."\n";
		exit;
	}
}
