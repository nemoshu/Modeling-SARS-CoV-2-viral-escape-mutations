"""
Converts a file in Clustal format to Newick format.

Command line arguments:
    path of input file (clustal), path of output file (newick)
"""

import sys

newick = ''

with open(sys.argv[1]) as f:
    for line in f:
        fields = line.rstrip().split(':')
        if len(fields) == 1:
            newick += fields[0] # no branch
        else:
            prefix, suffix = fields
            newick += prefix[:30]
            if suffix != '':
                newick += ':'
                newick += '{:.10f}'.format(abs(float(suffix))) # newick values

with open(sys.argv[2], 'w') as of:
    of.write(newick + '\n')
