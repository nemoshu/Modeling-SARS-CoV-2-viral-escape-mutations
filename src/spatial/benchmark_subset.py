from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
import sys


def msa_subset(ifname, ofname, anchor_id, cutoff=0):
    """
    Filters pre-aligned sequences based on gap consistency with anchor.

    Args:
        ifname (string): Name of the input file for the input FASTA (pre-aligned)
        ofname (string): Name of the output file for the output FASTA
        anchor_id (string): Reference sequence ID
        cutoff (int, optional): Maximum number of positions where the anchor has non-map AA and sequence has a gap.

    Returns:
        The processed anchor sequence as a String
    """

    # Load a FASTA alignment
    align = AlignIO.read(ifname, 'fasta')

    # Identify an anchor sequence
    anchor_idx = [ align[i].id
                   for i in range(len(align)) ].index(anchor_id)
    anchor = align[anchor_idx]

    # Retain sequences with non-gap mismatches no greater than the cutoff parameter, relative to the anchor
    subset = []
    for idx, record in enumerate(align):
        if idx % 10000 == 9999:
            print('Record {}...'.format(idx + 1))
        n_diff = sum([
            (x1 != '-' and x2 == '-')
            for x1, x2 in zip(anchor, align[idx])
        ])
        if n_diff <= cutoff:
            subset.append(align[idx])

    print('Found {} records'.format(len(subset)))

    align_subset = MultipleSeqAlignment(subset)
    AlignIO.write(align_subset, ofname, 'fasta')

    return str(anchor.seq)


def create_mutants(aligned_str):
    """
    Generates all possible single-amino-acid mutants of the anchor sequence.

    Args:
        aligned_str (string): An aligned anchor sequence

    Returns:
        mutants (list): A list of mutant sequences
        mutant_names (list): A list of corresponding mutant names, in the format "mut_<position mutated>_<amino acid mutated to>"
    """

    AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V', 'U',
    ] # possible amino acids

    mutants, mutant_names = [], []
    for i in range(len(aligned_str)):
        if aligned_str[i] == '-':
            continue # skip if character at position i is not an AA
        for aa in AAs:
            if aligned_str[i] == aa:
                continue # skip if mutation has no effect
            name = 'mut_{}_{}'.format(i, aa) # name format mut_<position mutated>_<amino acid mutated to>
            mutable = aligned_str[:i] + aa + aligned_str[i + 1:]
            mutant_names.append(name)
            mutants.append(mutable)
    return mutants, mutant_names

def write_mutants(mutants, mutant_names, outfile):
    """
    Writes generated mutants to a FASTA file.

    Args:
        mutants (list): A list of mutant sequences
        mutant_names (list): A list of corresponding mutant names
        outfile (string): Name of the output file
    """
    with open(outfile, 'w') as of:
        for mutant, name in zip(mutants, mutant_names):
            of.write('>{}\n'.format(name))
            of.write('{}\n'.format(mutant))

if __name__ == '__main__':

    """
    Splits into subsets, and creates mutations for each dataset.
    
    Input Files: 
        - target/influenza/clusters/all.fasta
        - target/hiv/clusters/all.fasta
        - target/cov/clusters/all.fasta
    
    Output Files:
        - target/influenza/clusters/all_h1.fasta (H1)
        - target/influenza/mutation/mutations_h1.fa (H1 mutations)
        - target/influenza/clusters/all_h3.fasta (H3)
        - target/influenza/mutation/mutations_h3.fa (H3 mutations)
        - target/hiv/clusters/all_BG505.fasta (HIV BG505)
        - target/hiv/mutation/mutations_hiv.fa (HIV BG505 mutations)
        - target/hiv/clusters/all_BF520.fasta (HIV BF520)
        - target/hiv/mutation/mutations_bf520.fa (HIV BF520 mutations)
        - target/cov/clusters/all_sarscov2.fasta (SARSCOV2)
        - target/cov/mutation/mutations_sarscov2.fa (SARSCOV2 mutations)
    """


    print('H1...')
    anchor = msa_subset(
        'target/influenza/clusters/all.fasta',
        'target/influenza/clusters/all_h1.fasta',
        'gb:LC333185|ncbiId:BBB04702.1|UniProtKB:-N/A-|'
        'Organism:Influenza', 2
    )
    mutants, mutant_names = create_mutants(anchor)
    write_mutants(mutants, mutant_names,
                  'target/influenza/mutation/mutations_h1.fa')

    print('H3...')
    anchor = msa_subset(
        'target/influenza/clusters/all.fasta',
        'target/influenza/clusters/all_h3.fasta',
        'Reference_Perth2009_HA_coding_sequence', 0
    )
    mutants, mutant_names = create_mutants(anchor)
    write_mutants(mutants, mutant_names,
                  'target/influenza/mutation/mutations_h3.fa')

    print('HIV BG505...')
    anchor = msa_subset(
        'target/hiv/clusters/all.fasta',
        'target/hiv/clusters/all_BG505.fasta',
        'A1.KE.-.BG505_W6M_ENV_C2.DQ208458', 15
    )
    mutants, mutant_names = create_mutants(anchor)
    write_mutants(mutants, mutant_names,
                  'target/hiv/mutation/mutations_hiv.fa')

    print('HIV BF520...')
    anchor = msa_subset(
        'target/hiv/clusters/all.fasta',
        'target/hiv/clusters/all_BF520.fasta',
        'A1.KE.1994.BF520.W14M.C2.KX168094', 15
    )
    mutants, mutant_names = create_mutants(anchor)
    write_mutants(mutants, mutant_names,
                  'target/hiv/mutation/mutations_bf520.fa')

    print('SARS-CoV-2...')
    anchor = msa_subset(
        'target/cov/clusters/all.fasta',
        'target/cov/clusters/all_sarscov2.fasta',
        'YP_009724390.1', 0
    )
    mutants, mutant_names = create_mutants(anchor)
    write_mutants(mutants, mutant_names,
                  'target/cov/mutation/mutations_sarscov2.fa')
