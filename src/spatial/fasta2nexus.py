"""
Converts FASTA format to NEXUS format.
Also generates a second file compatible with MrBayes under '{output file name} + mrbayes.nex'.

Command line arguments:
    path of input file (FASTA), path of output file (NEXUS)
"""

from Bio import Align, AlignIO, SeqIO, Nexus, Seq
from Bio.Alphabet import IUPAC, Gapped
import sys

seq_set = set()

records = AlignIO.read(sys.argv[1], 'fasta', alphabet=Gapped(IUPAC.protein))
new_records = []
for idx in range(len(records)):
    # replace ambiguous AAs
    seq = str(records[idx].seq).replace('J', 'L').replace('B', 'N').replace('Z', 'Q')
    # remove duplicates
    if seq in seq_set:
        continue
    seq_set.add(seq)
    record = records[idx]
    # standardize ID format
    record.id = record.id[:99].replace('-', '_')
    record.seq = Seq.Seq(seq)
    new_records.append(record)
new_records = Align.MultipleSeqAlignment(new_records, alphabet=Gapped(IUPAC.protein))

with open(sys.argv[2], 'w') as of:
    AlignIO.write(new_records, of, 'nexus')

# generate a new output file supporting mrbayes
with open(sys.argv[2], 'r') as f:
    nex = Nexus.Nexus.Nexus()
    nex.read(f)

with open(sys.argv[2] + 'mrbayes.nex', 'w') as of:
    nex.write_nexus_data(of, mrbayes=True)
