#!/usr/bin/env perl
if($#ARGV < 0){
	die "Usage: ReadInput.pl <filename> <is_sparse>\n  <filename>: the file in MILL format \n <is_sparse>: 0 (default), 1\n"	 
}

my $data_file = $ARGV[0];
my $is_sparse = 0;
if($#ARGV >= 1){
	if($ARGV[1] != 0){
		$is_sparse = 1;	
	}
}

open IN, "<$data_file" or die "Cannot open input file $data_file for reading\n";
my @lines = <IN>;
chomp(@lines);
close IN;
	
my $matrix_file = $data_file.".matrix";
my $label_file = $data_file.".label";
open OUT_MAT, ">$matrix_file" or die "Cannot open matrix file $matrix_file for writing.\n";
open OUT_LABEL, ">$label_file" or die "Cannot open label file $label_file for writing.\n";

my $data_id = 1;
foreach my $line(@lines)
{
	last if(length($line) == 0);
	next if($line =~ /^#/);	
		
	my @elems = split(/[, \t]/, $line);
	
	#write the labels
	my $inst_name = $elems[0];
	my $bag_name = $elems[1];
	my $label = $elems[2];
	if($label == -1){
		$label = 0;	
	}		
	print OUT_LABEL $inst_name, " ", $bag_name, " ", $label, "\n"; 	
	
	#write the data
	for(my $i = 3; $i <= $#elems; $i = $i + 1)
	{
		if($is_sparse == 1)
		{
			my @vals = split(/:/, $elems[$i]);
			if($#vals != 1){
				die "Corrupted feature $elems[$i] in line $data_id\n";
			}
			print OUT_MAT $data_id, " ", $vals[0], " ", $vals[1], "\n"; 
		}
		else
		{
			print OUT_MAT $elems[$i], " "; 
		}		
	}
	if($is_sparse == 0){
		print OUT_MAT "\n"; 
	}
	$data_id = $data_id + 1;
}
close OUT_MAT;
close OUT_LABEL;
