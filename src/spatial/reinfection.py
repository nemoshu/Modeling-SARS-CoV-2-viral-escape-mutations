from utils import *

def load_to2020():
    """
    Loads the To et al. 2020 reinfection data.

    Returns:
        seq (string): the original sequence
        mutants (dict[int: list[string]]: {4: [4-mutation sequence]}
    """
    seq = str(SeqIO.read('data/cov/cov2_spike_wt.fasta', 'fasta').seq)
    seq = seq[:779] + 'Q' + seq[780:]

    muts = [
        [ 'L18F', 'A222V', 'D614G', 'Q780E' ],
    ]

    mutants = { 4: [] } # maps number of mutations to a list of sequences with that number of mutations
    for i in range(len(muts)):
        assert(len(muts[i]) == 4) # 4-mutation sequences only
        mutable = seq
        for mut in muts[i]:
            aa_orig = mut[0]
            aa_mut = mut[-1]
            pos = int(mut[1:-1]) - 1
            assert(seq[pos] == aa_orig)
            mutable = mutable[:pos] + aa_mut + mutable[pos + 1:]
        mutants[4].append(mutable)

    return seq, mutants

def load_ratg13():
    """
    Loads the RaTG13 reinfection data.

    Returns:
        seq (string): the original sequence
        mutants (dict[int: list[string]]: {8: [8-mutation sequence]})
    """
    seq = str(SeqIO.read('data/cov/cov2_spike_wt.fasta', 'fasta').seq)

    muts = [
        [ 'N439K', 'Y449F', 'F486L', 'Q493Y',
          'S494R', 'Q498Y', 'N501D', 'Y505H' ],
    ]

    mutants = { 8: [] } # maps number of mutations to a list of sequences with that number of mutations
    for i in range(len(muts)):
        assert(len(muts[i]) == 8) # 8-mutation sequences only
        mutable = seq
        for mut in muts[i]:
            aa_orig = mut[0]
            aa_mut = mut[-1]
            pos = int(mut[1:-1]) - 1
            assert(seq[pos] == aa_orig)
            mutable = mutable[:pos] + aa_mut + mutable[pos + 1:]
        mutants[8].append(mutable)

    return seq, mutants

def load_sarscov1():
    """
    Loads the SARS-CoV-1 cross-species case.

    Returns:
        seq (string): the original sequence
        mutants (dict[int: list[string]]: {12: [12-mutation sequence]})
    """
    seq = str(SeqIO.read('data/cov/cov2_spike_wt.fasta', 'fasta').seq)

    muts = [
        [ 'K417V', 'N439R', 'G446T', 'L455Y',
          'F456L', 'A475P', 'F486L', 'Q493N',
          'S494D', 'Q498Y', 'N501T', 'V503I' ],
    ]

    mutants = { 12: [] } # maps number of mutations to a list of sequences with that number of mutations
    for i in range(len(muts)):
        assert(len(muts[i]) == 12) # 12-mutation sequences onlyfrom utils import *

def load_to2020():
    """
    Loads the To et al. 2020 reinfection data.

    Returns:
        seq (string): the original sequence
        mutants (dict[int: list[string]]: {4: [4-mutation sequence]}
    """
    seq = str(SeqIO.read('data/cov/cov2_spike_wt.fasta', 'fasta').seq)
    seq = seq[:779] + 'Q' + seq[780:]

    muts = [
        [ 'L18F', 'A222V', 'D614G', 'Q780E' ],
    ]

    mutants = { 4: [] } # maps number of mutations to a list of sequences with that number of mutations
    for i in range(len(muts)):
        assert(len(muts[i]) == 4) # 4-mutation sequences only
        mutable = seq
        for mut in muts[i]:
            aa_orig = mut[0]
            aa_mut = mut[-1]
            pos = int(mut[1:-1]) - 1
            assert(seq[pos] == aa_orig)
            mutable = mutable[:pos] + aa_mut + mutable[pos + 1:]
        mutants[4].append(mutable)

    return seq, mutants

def load_ratg13():
    """
    Loads the RaTG13 reinfection data.

    Returns:
        seq (string): the original sequence
        mutants (dict[int: list[string]]: {8: [8-mutation sequence]})
    """
    seq = str(SeqIO.read('data/cov/cov2_spike_wt.fasta', 'fasta').seq)

    muts = [
        [ 'N439K', 'Y449F', 'F486L', 'Q493Y',
          'S494R', 'Q498Y', 'N501D', 'Y505H' ],
    ]

    mutants = { 8: [] } # maps number of mutations to a list of sequences with that number of mutations
    for i in range(len(muts)):
        assert(len(muts[i]) == 8) # 8-mutation sequences only
        mutable = seq
        for mut in muts[i]:
            aa_orig = mut[0]
            aa_mut = mut[-1]
            pos = int(mut[1:-1]) - 1
            assert(seq[pos] == aa_orig)
            mutable = mutable[:pos] + aa_mut + mutable[pos + 1:]
        mutants[8].append(mutable)

    return seq, mutants

def load_sarscov1():
    """
    Loads the SARS-CoV-1 cross-species case.

    Returns:
        seq (string): the original sequence
        mutants (dict[int: list[string]]: {12: [12-mutation sequence]})
    """
    seq = str(SeqIO.read('data/cov/cov2_spike_wt.fasta', 'fasta').seq)

    muts = [
        [ 'K417V', 'N439R', 'G446T', 'L455Y',
          'F456L', 'A475P', 'F486L', 'Q493N',
          'S494D', 'Q498Y', 'N501T', 'V503I' ],
    ]

    mutants = { 12: [] } # maps number of mutations to a list of sequences with that number of mutations
    for i in range(len(muts)):
        assert(len(muts[i]) == 12) # 12-mutation sequences only
        mutable = seq
        for mut in muts[i]:
            aa_orig = mut[0]
            aa_mut = mut[-1]
            pos = int(mut[1:-1]) - 1
            assert(seq[pos] == aa_orig)
            mutable = mutable[:pos] + aa_mut + mutable[pos + 1:]
        mutants[12].append(mutable)

    return seq, mutants

if __name__ == '__main__':
    """
    Testing purposes only. No data processing to be conducted.
    """
    load_to2020()
    load_ratg13()
    load_sarscov1()