from utils import Counter, SeqIO

from Bio.Seq import translate
import numpy as np

def load_doud2016():
    """
    Loads H1 influenza fitness data.

    Returns:
        - strains (dict): Dictionary of wild-type sequences {strain: wt_seq}
        - seqs_fitness (dict): Dictionary of mutation data, keyed by (mut_seq, strain), containing the following data:
            - fitnesses
            - preferences
            - wildtype
            - mut_pos
    """
    strain = 'h1'

    fname = 'data/influenza/escape_doud2018/WSN1933_H1_HA.fa'
    wt_seq = SeqIO.read(fname, 'fasta').seq # wild-type sequence

    seqs_fitness = {}
    fname = \
        ('data/influenza/fitness_doud2016/'
             'Supplemental_File_2_HApreferences.txt')
    with open(fname) as f:
        muts = f.readline().rstrip().split()[4:] # mutants
        for line in f:
            fields = line.rstrip().split() # data fields available
            pos = int(fields[0]) - 1
            orig = fields[1]
            assert(wt_seq[pos] == orig) # confirm the reference WT matches
            data = [ float(field) for field in fields[3:] ] # all remaining columns
            assert(len(muts) == len(data)) # confirm mutation headers matches actual data
            for mut, pref in zip(muts, data):
                mutable = [ aa for aa in wt_seq ] # list of characters from wt_seq
                assert(mut.startswith('PI_'))
                mutable[pos] = mut[-1]
                mut_seq = ''.join(mutable) # mutated sequence
                assert(len(mut_seq) == len(wt_seq)) # confirm mutated sequence is the same length
                if (mut_seq, strain) not in seqs_fitness:
                    seqs_fitness[(mut_seq, strain)] = [ {
                        'strain': strain,
                        'fitnesses': [ pref ],
                        'preferences': [ pref ],
                        'wildtype': wt_seq,
                        'mut_pos': [ pos ],
                    } ]
                else:
                    seqs_fitness[(mut_seq, strain)][0][
                        'fitnesses'].append(pref)
                    seqs_fitness[(mut_seq, strain)][0][
                        'preferences'].append(pref)

    # compute medians
    for fit_key in seqs_fitness:
        seqs_fitness[fit_key][0]['fitness'] = np.median(
            seqs_fitness[fit_key][0]['fitnesses']
        )
        seqs_fitness[fit_key][0]['preference'] = np.median(
            seqs_fitness[fit_key][0]['preferences']
        )

    return { strain: wt_seq }, seqs_fitness

def load_haddox2018():
    """
    Loads HIV fitness data.

    Input Files:
        - data/hiv/fitness_haddox2018/BF520_env.fasta
        - data/hiv/fitness_haddox2018/BF520_to_HXB2.csv
        - data/hiv/fitness_haddox2018/BF520_avgprefs.csv
        - data/hiv/fitness_haddox2018/BG505_env.fasta
        - data/hiv/fitness_haddox2018/BG505_to_HXB2.csv
        - data/hiv/fitness_haddox2018/BG505_avgprefs.csv

    Returns:
        - strains (dict): Dictionary of wild-type sequences {strain: wt_seq}
        - seqs_fitness (dict): Dictionary of mutation data, keyed by (mut_seq, strain), containing the following data:
            - strain
            - fitnesses
            - preferences
            - wildtype
            - mut_pos
    """
    strain_names = [ 'BF520', 'BG505' ]

    strains = {}  # maps strains to sequence
    seqs_fitness = {}  # maps strains to metadata
    for strain in strain_names:
        wt_seq = translate(SeqIO.read(
            'data/hiv/fitness_haddox2018/'
            '{}_env.fasta'.format(strain), 'fasta'
        ).seq).rstrip('*')
        strains[strain] = wt_seq

        fname = 'data/hiv/fitness_haddox2018/{}_to_HXB2.csv'.format(strain)
        pos_map = {} # map from sequence to (origin, position)
        with open(fname) as f:
            f.readline() # Consume header.
            for line in f:
                fields = line.rstrip().split(',')
                pos_map[fields[1]] = (fields[2], int(fields[0]) - 1)

        fname = ('data/hiv/fitness_haddox2018/{}_avgprefs.csv'
                 .format(strain))
        with open(fname) as f:
            mutants = f.readline().rstrip().split(',')[1:]
            for line in f:
                fields = line.rstrip().split(',')
                orig, pos = pos_map[fields[0]]
                assert(wt_seq[int(pos)] == orig) # confirm that wt_sequence at position matches the original
                preferences = [ float(field) for field in fields[1:] ]
                assert(len(mutants) == len(preferences)) # confirm that each mutant matches has a preference
                for mut, pref in zip(mutants, preferences):
                    # insert mutation
                    mutable = [ aa for aa in wt_seq ]
                    mutable[pos] = mut
                    mut_seq = ''.join(mutable)
                    if (mut_seq, strain) not in seqs_fitness:
                        seqs_fitness[(mut_seq, strain)] = [ {
                            'strain': strain,
                            'fitnesses': [ pref ],
                            'preferences': [ pref ],
                            'wildtype': wt_seq,
                            'mut_pos': [ pos ],
                        } ]
                    else:
                        seqs_fitness[(mut_seq, strain)][0][
                            'fitnesses'].append(pref)
                        seqs_fitness[(mut_seq, strain)][0][
                            'preferences'].append(pref)

    # compute and add medians
    for fit_key in seqs_fitness:
        seqs_fitness[fit_key][0]['fitness'] = np.median(
            seqs_fitness[fit_key][0]['fitnesses']
        )
        seqs_fitness[fit_key][0]['preference'] = np.median(
            seqs_fitness[fit_key][0]['preferences']
        )

    return strains, seqs_fitness

def load_wu2020():
    """
    Loads combinatorial mutation data.

    Input Files:
        - data/influenza/fitness_wu2020/wildtypes.fa
        - data/influenza/fitness_wu2020/data_pref.tsv


    Returns:
        - strains (dict): Dictionary of wild-type sequences {strain: wt_seq}
        - seqs_fitness (dict): Dictionary of mutation data, keyed by (mut_seq, strain), containing the following data:
            - fitness
            - preference
            - wildtype
            - mut_pos
    """
    mut_pos = [
        156, 158, 159, 190, 193, 196
    ]
    offset = 16 # Amino acids in prefix.
    mut_pos = [ pos - 1 + offset for pos in mut_pos ]

    names = [
        'HK68', 'Bk79', 'Bei89', 'Mos99', 'Bris07L194', 'NDako16',
    ]
    wildtypes = [
        'KGSESV', 'EESENV', 'EEYENV', 'QKYDST', 'HKFDFA', 'HNSDFA',
    ]

    # Load full wildtype sequences.

    wt_seqs = {}
    fname = 'data/influenza/fitness_wu2020/wildtypes.fa'
    for record in SeqIO.parse(fname, 'fasta'):
        strain_idx = names.index(record.description)
        wt = wildtypes[strain_idx]
        for aa, pos in zip(wt, mut_pos):
            assert(record.seq[pos] == aa) # check if record sequence matches the wild type
        wt_seqs[names[strain_idx]] = record.seq

    # Load mutants.

    seqs_fitness = {}
    fname = 'data/influenza/fitness_wu2020/data_pref.tsv'
    with open(fname) as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split('\t')
            mut, strain, fitness, preference = fields
            if strain == 'Bris07P194':
                continue
            if strain == 'Bris07':
                strain = 'Bris07L194'
            fitness = float(preference)
            preference = float(preference)

            strain_idx = names.index(strain)
            wt = wildtypes[strain_idx]
            full_seq = wt_seqs[strain]

            mutable = [ aa for aa in full_seq ]
            # insert mutation
            for aa_wt, aa, pos in zip(wt, mut, mut_pos):
                assert(mutable[pos] == aa_wt)
                mutable[pos] = aa
            mut_seq = ''.join(mutable) # convert to flat string

            if (mut_seq, strain) not in seqs_fitness:
                seqs_fitness[(mut_seq, strain)] = []
            seqs_fitness[(mut_seq, strain)].append({
                'strain': strain,
                'fitness': fitness,
                'preference': preference,
                'wildtype': full_seq,
                'mut_pos': mut_pos,
            })

    return wt_seqs, seqs_fitness

def load_starr2020():
    """
    Loads SARS-CoV-2 binding data.

    Input files:
        - data/cov/cov2_spike_wt.fasta
        - data/cov/starr2020cov2/binding_Kds.csv

    Returns:
        - strains (dict): Dictionary of wild-type sequences {strain: wt_seq}
        - seqs_fitness (dict): Dictionary of mutation data, keyed by (mut_seq, strain), containing the following data:
            - fitnesses
            - preferences
            - wildtype
            - mut_pos
    """

    strain = 'sars_cov_2'
    wt_seq = SeqIO.read('data/cov/cov2_spike_wt.fasta', 'fasta').seq

    seqs_fitness = {}
    with open('data/cov/starr2020cov2/binding_Kds.csv') as f:
        f.readline()
        for line in f:
            fields = line.replace('"', '').rstrip().split(',')
            if fields[5] == 'NA':
                continue
            log10Ka = float(fields[5])
            mutants = fields[-2].split()
            mutable = [ aa for aa in wt_seq ]
            mut_pos = []
            for mutant in mutants:
                orig, mut = mutant[0], mutant[-1]
                pos = int(mutant[1:-1]) - 1 + 330
                assert(wt_seq[pos] == orig) # confirm wild-type sequence matches original
                mutable[pos] = mut # perform mutation
                mut_pos.append(pos)
            mut_seq = ''.join(mutable)

            if (mut_seq, strain) not in seqs_fitness:
                seqs_fitness[(mut_seq, strain)] = [ {
                    'strain': strain,
                    'fitnesses': [ log10Ka ],
                    'preferences': [ log10Ka ],
                    'wildtype': wt_seq,
                    'mut_pos': mut_pos,
                } ]
            else:
                seqs_fitness[(mut_seq, strain)][0][
                    'fitnesses'].append(log10Ka)
                seqs_fitness[(mut_seq, strain)][0][
                    'preferences'].append(log10Ka)

    # calculate medians
    for fit_key in seqs_fitness:
        seqs_fitness[fit_key][0]['fitness'] = np.median(
            seqs_fitness[fit_key][0]['fitnesses']
        )
        seqs_fitness[fit_key][0]['preference'] = np.median(
            seqs_fitness[fit_key][0]['preferences']
        )

    print(len(seqs_fitness))

    return { strain: wt_seq }, seqs_fitness

if __name__ == '__main__':
    """
    For test purposes only. Loaded data is not processed.
    """
    load_starr2020()
    load_doud2016()
    load_haddox2018()
    load_wu2020()
