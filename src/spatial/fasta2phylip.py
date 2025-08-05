"""
Converts FASTA format to PHYLIP format.

Command line arguments:
    path of input file (FASTA), path of output file (NEXUS)
"""

from Bio import Align, AlignIO, Seq
import sys

seq_set = set()

records = AlignIO.read(sys.argv[1], 'fasta')
new_records = []
for idx in range(len(records)):
    # resolve ambiguity
    seq = str(records[idx].seq).replace('J', 'L').replace('B', 'N').replace('Z', 'Q')
    # ignore duplicates
    if seq in seq_set:
        continue
    seq_set.add(seq)
    record = records[idx]
    record.seq = Seq.Seq(seq)
    new_records.append(record)
new_records = Align.MultipleSeqAlignment(new_records)

with open(sys.argv[2], 'w') as of:
    AlignIO.PhylipIO.PhylipWriter(of).write_alignment(
        new_records, id_width=100
    )
