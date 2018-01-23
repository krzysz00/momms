#!/bin/bash
cut -f 1,5-6 $1 | perl -ne 'BEGIN {my $sum = 0; my $iter = 0; } { next if /^\s*#/; my ($n, $o, $t) = split "\t"; my $diff = ($o - $t)/$t; print "$n $diff\n"; $sum += $diff; $iter += 1; } END { my $avg = $sum/$iter; print "# $avg\n"; }'
