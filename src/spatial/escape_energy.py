from utils import *

from sklearn.metrics import auc

def parse_args():
    """
    Parses command line arguments

    Returns:
        argparse.Namespace: the parsed arguments
    """
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark methods')
    parser.add_argument('method', type=str,
                        help='Benchmarking method to use')
    parser.add_argument('virus', type=str,
                        help='Viral type to test')
    args = parser.parse_args()
    return args

def load(virus):
    """
    Loads the sequences and file paths relevant to the specified virus.

    Args:
        virus (str): the name of the virus to load

    Returns:
        tuple:
            seq (string): Wild-type reference sequence
            seqs_escape (dict[str, list[dict]]): the dictionary which maps mutant sequence to escape metadata
            train_fname (str): path to the training file
            mut_fname (str): path to all single-mutation sequences
            anchor_id (str): FASTA ID of the wild-type
    """
    if virus == 'h1':
        from escape import load_doud2018
        seq, seqs_escape = load_doud2018()
        train_fname = 'target/influenza/clusters/all_h1.fasta'
        mut_fname = 'target/influenza/mutation/mutations_h1.fa'
        anchor_id = ('gb:LC333185|ncbiId:BBB04702.1|UniProtKB:-N/A-|'
                     'Organism:Influenza')
    elif virus == 'h3':
        from escape import load_lee2019
        seq, seqs_escape = load_lee2019()
        train_fname = 'target/influenza/clusters/all_h3.fasta'
        mut_fname = 'target/influenza/mutation/mutations_h3.fa'
        anchor_id = 'Reference_Perth2009_HA_coding_sequence'
    elif virus == 'bg505':
        from escape import load_dingens2019
        seq, seqs_escape = load_dingens2019()
        train_fname = 'target/hiv/clusters/all_BG505.fasta'
        mut_fname = 'target/hiv/mutation/mutations_hiv.fa'
        anchor_id = 'A1.KE.-.BG505_W6M_ENV_C2.DQ208458'
    elif virus == 'sarscov2':
        from escape import load_baum2020
        seq, seqs_escape = load_baum2020()
        train_fname = 'target/cov/clusters/all_sarscov2.fasta'
        mut_fname = 'target/cov/mutation/mutations_sarscov2.fa'
        anchor_id = 'YP_009724390.1'
    elif virus == 'cov2rbd':
        from escape import load_greaney2020
        seq, seqs_escape = load_greaney2020()
        train_fname = 'target/cov/clusters/all_sarscov2.fasta'
        mut_fname = 'target/cov/mutation/mutations_sarscov2.fa'
        anchor_id = 'YP_009724390.1'
    else:
        raise ValueError('invalid option {}'.format(virus))

    return seq, seqs_escape, train_fname, mut_fname, anchor_id

def plot_result(rank_vals, escape_idx, virus, fname_prefix,
                legend_name='Result'):
    """
    Plots and saves cumulative hit curves for top ranked mutations and computes AUC.

    Args:
        rank_vals(nparray): ranking for mutations by escape potential
        escape_idx(list): indices for known escape mutants
        virus (str): virus name
        fname_prefix(str): prefix for filenames
        legend_name(str): label for legend

    Returns:
        None
    """
    acq_argsort = ss.rankdata(-rank_vals)
    escape_rank_dist = acq_argsort[escape_idx]

    n_consider = np.array([ i + 1 for i in range(len(rank_vals)) ]) # ranks list

    n_escape = np.array([ sum(escape_rank_dist <= i + 1)
                          for i in range(len(rank_vals)) ]) # cumulative escape
    norm = max(n_consider) * max(n_escape) # normalizer
    norm_auc = auc(n_consider, n_escape) / norm # normalized AUC

    escape_frac = len(escape_rank_dist) / float(len(rank_vals))

    tprint('Results for {} ({}):'.format(virus, legend_name))
    tprint('AUC = {}'.format(norm_auc))

    plt.figure()
    plt.plot(n_consider, n_escape)
    plt.plot(n_consider, n_consider * escape_frac,
             c='gray', linestyle='--')
    plt.legend([
        r'{}, AUC = {:.3f}'.format(legend_name, norm_auc),
        'Random guessing, AUC = 0.500'
    ])
    plt.xlabel('Top N')
    plt.ylabel('Number of escape mutations in top N')
    plt.savefig('figures/{}_{}_escape.png'
                .format(virus, fname_prefix), dpi=300)
    plt.close()

def escape_evcouplings(virus, vocabulary):
    """
    Evaluates epistatic and independent EVCoupling scores for each mutation and benchmark, and plots results.

    Args:
        virus (str): virus name
        vocabulary(list[str]): amino acids allowed in mutations

    Returns:
        None
    """
    seq, seqs_escape, train_fname, mut_fname, anchor_id = load(virus)

    # decide energy file name
    if virus == 'h1':
        energy_fname = ('target/influenza/evcouplings/flu_h1/mutate/'
                        'flu_h1_single_mutant_matrix.csv')
    elif virus == 'h3':
        energy_fname = ('target/influenza/evcouplings/flu_h3/mutate/'
                        'flu_h3_single_mutant_matrix.csv')
    elif virus == 'bg505':
        energy_fname = ('target/hiv/evcouplings/hiv_env/mutate/'
                        'hiv_env_single_mutant_matrix.csv')
    elif virus == 'sarscov2':
        energy_fname = ('target/cov/evcouplings/sarscov2/mutate/'
                        'sarscov2_single_mutant_matrix.csv')
    elif virus == 'cov2rbd':
        energy_fname = ('target/cov/evcouplings/sarscov2/mutate/'
                        'sarscov2_single_mutant_matrix.csv')
    else:
        raise ValueError('invalid option {}'.format(virus))

    anchor = None
    for idx, record in enumerate(SeqIO.parse(train_fname, 'fasta')):
        if record.id == anchor_id:
            anchor = str(record.seq).replace('-', '')
    assert(anchor is not None) # an anchor should be found in training file

    pos_aa_score_epi = {} # maps (pos, mut) to epistemic score
    pos_aa_score_ind = {} # maps (pos, mut) to independent score
    with open(energy_fname) as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split(',')
            pos = int(fields[2]) - 1
            orig, mut = fields[3], fields[4]
            assert(anchor[pos] == orig)
            pos_aa_score_epi[(pos, mut)] = float(fields[7])
            pos_aa_score_ind[(pos, mut)] = float(fields[8])

    escape_idx = []
    mut_scores_epi, mut_scores_ind = [], []
    for i in range(len(anchor)):
        if virus == 'bg505' and (i < 29 or i > 698):
            continue
        if virus == 'cov2rbd' and (i < 318 or i > 540):
            continue
        for word in vocabulary:
            if anchor[i] == word:
                continue # skip repeats
            mut_seq = anchor[:i] + word + anchor[i + 1:]
            if mut_seq in seqs_escape and \
               (sum([ m['significant'] for m in seqs_escape[mut_seq] ]) > 0):
                escape_idx.append(len(mut_scores_epi))
            if (i, word) in pos_aa_score_epi:
                mut_scores_epi.append(pos_aa_score_epi[(i, word)])
                mut_scores_ind.append(pos_aa_score_ind[(i, word)])
            else:
                mut_scores_epi.append(0)
                mut_scores_ind.append(0)
    mut_scores_epi = np.array(mut_scores_epi)
    mut_scores_ind = np.array(mut_scores_ind)

    plot_result(mut_scores_epi, escape_idx, virus, 'evcouplings_epi',
                legend_name='EVcouplings (epistatic)')
    plot_result(mut_scores_ind, escape_idx, virus, 'evcouplings_ind',
                legend_name='EVcouplings (independent)')

def escape_freq(virus, vocabulary):
    """
    Benchmarks prediction of escape mutations by mutation frequency, and plots the results

    Args:
        virus (string): the name of the virus
        vocabulary (list[string]): amino acids allowed in mutations
    Returns:
        None
    """
    seq, seqs_escape, train_fname, mut_fname, anchor_id = load(virus)

    anchor = None
    pos_aa_freq = {} # maps (position, character) to frequency
    for idx, record in enumerate(SeqIO.parse(train_fname, 'fasta')):
        mutation = str(record.seq)
        if record.id == anchor_id:
            anchor = mutation
        else:
            for pos, c in enumerate(mutation):
                # count presence
                if c == '-':
                    continue # skip non-mutation
                if (pos, c) not in pos_aa_freq:
                    pos_aa_freq[(pos, c)] = 0.
                pos_aa_freq[(pos, c)] += 1.
    assert(anchor is not None) # an anchor must be found somewhere in the record

    escape_idx, mut_freqs = [], []
    real_pos = 0 # real position tracker (ignoring '-'s)
    for i in range(len(anchor)):
        if anchor[i] == '-':
            continue
        if virus == 'bg505' and (real_pos < 29 or real_pos > 698):
            real_pos += 1
            continue
        if virus == 'cov2rbd' and (real_pos < 318 or real_pos > 540):
            real_pos += 1
            continue
        for word in vocabulary:
            if anchor[i] == word:
                continue # skip mutations without effect
            mut_seq = anchor[:i] + word + anchor[i + 1:]
            mut_seq = mut_seq.replace('-', '') # perform mutation and remove ambiguity
            if mut_seq in seqs_escape and \
               (sum([ m['significant'] for m in seqs_escape[mut_seq] ]) > 0):
                escape_idx.append(len(mut_freqs))
            if (i, word) in pos_aa_freq:
                mut_freqs.append(pos_aa_freq[(i, word)])
            else:
                mut_freqs.append(0)
        real_pos += 1
    mut_freqs = np.array(mut_freqs) # convert to nparray for plotting

    plot_result(mut_freqs, escape_idx, virus, 'mutfreq',
                legend_name='Mutation frequency')

def tape_embed(sequence, model, tokenizer):
    """
    Computes mean embedding using TAPE model.

    Args:
        sequence (str): the sequence to be embedded
        model (torch.nn.Module): the model to be used
        tokenizer (Tokenizer): TAPE tokenizer

    Returns:
        np.ndarray: the mean embedding vector
    """
    import torch
    token_ids = torch.tensor([tokenizer.encode(sequence)])
    output = model(token_ids)
    return output[0].detach().numpy().mean(1).ravel()

def escape_tape(virus, vocabulary, pretrained='transformer'):
    """
    Benchmarks escape prediction using TAPE model.

    Args:
        virus (string): the name of the virus
        vocabulary (list[string]): amino acids allowed in mutations
        pretrained (string): the pretrained model ('transformer' or 'unirep')

    Returns:
        None
    """
    if pretrained == 'transformer':
        from tape import ProteinBertModel
        model_class = ProteinBertModel
        model_name = 'bert-base'
        fname_prefix = 'tape_transformer'
        vocab = 'iupac'
    elif pretrained == 'unirep':
        from tape import UniRepModel
        model_class = UniRepModel
        model_name = 'babbler-1900'
        fname_prefix = 'unirep'
        vocab = 'unirep'

    from tape import TAPETokenizer
    model = model_class.from_pretrained(model_name)
    tokenizer = TAPETokenizer(vocab=vocab)

    seq, seqs_escape, train_fname, mut_fname, anchor_id = load(virus)
    if virus == 'h1':
        embed_fname = ('target/influenza/embedding/{}_h1.npz'
                       .format(fname_prefix))
    elif virus == 'h3':
        embed_fname = ('target/influenza/embedding/{}_h3.npz'
                       .format(fname_prefix))
    elif virus == 'bg505':
        embed_fname = ('target/hiv/embedding/{}_hiv.npz'
                       .format(fname_prefix))
    elif virus == 'sarscov2':
        embed_fname = ('target/cov/embedding/{}_sarscov2.npz'
                       .format(fname_prefix))
    elif virus == 'cov2rbd':
        embed_fname = ('target/cov/embedding/{}_sarscov2.npz'
                       .format(fname_prefix))
    else:
        raise ValueError('invalid option {}'.format(virus))

    anchor = None
    for idx, record in enumerate(SeqIO.parse(train_fname, 'fasta')):
        if record.id == anchor_id:
            anchor = str(record.seq)
    assert(anchor is not None) # an anchor must be found in the record

    base_embedding = tape_embed(anchor.replace('-', ''),
                                model, tokenizer)
    with np.load(embed_fname, allow_pickle=True) as data:
        embeddings = { name: data[name][()]['avg']
                       for name in data.files }

    mutations = [
        str(record.seq) for record in SeqIO.parse(mut_fname, 'fasta')
    ]
    mut2change = {} # maps mutation to semantic change
    for mutation in mutations:
        didx = [ c1 != c2
                 for c1, c2 in zip(anchor, mutation) ].index(True) # first index of difference
        embedding = embeddings['mut_{}_{}'.format(didx, mutation[didx])] # embedding indexed by first mutation
        mutation_clean = mutation.replace('-', '') # remove ambiguities
        mut2change[mutation_clean] = abs(base_embedding - embedding).sum()

    anchor = anchor.replace('-', '') # remove ambiguities
    escape_idx, changes = [], []
    for i in range(len(anchor)):
        if virus == 'bg505' and (i < 29 or i > 698):
            continue
        if virus == 'cov2rbd' and (i < 318 or i > 540):
            continue
        for word in vocabulary:
            if anchor[i] == word:
                continue
            mut_seq = anchor[:i] + word + anchor[i + 1:]
            if mut_seq in seqs_escape and \
               (sum([ m['significant'] for m in seqs_escape[mut_seq] ]) > 0):
                escape_idx.append(len(changes))
            changes.append(mut2change[mut_seq])
    changes = np.array(changes) # map to nparray for results plot

    plot_result(-changes, escape_idx, virus, fname_prefix,
                legend_name='TAPE ({})'.format(fname_prefix))


def escape_bepler(virus, vocabulary):
    """
    Benchmarks escape prediction using Belper & Berger model.

    Args:
        virus (string): the name of the virus
        vocabulary (list[string]): amino acids allowed in mutations
        pretrained (string): the pretrained model ('transformer' or 'unirep')

    Returns:
        None
    """
    seq, seqs_escape, train_fname, mut_fname, anchor_id = load(virus)
    if virus == 'h1':
        embed_fname = 'target/influenza/embedding/bepler_ssa_h1.txt'
    elif virus == 'h3':
        embed_fname = 'target/influenza/embedding/bepler_ssa_h3.txt'
    elif virus == 'bg505':
        embed_fname = 'target/hiv/embedding/bepler_ssa_hiv.txt'
    elif virus == 'sarscov2':
        embed_fname = 'target/cov/embedding/bepler_ssa_sarscov2.txt'
    elif virus == 'cov2rbd':
        embed_fname = 'target/cov/embedding/bepler_ssa_sarscov2.txt'
    else:
        raise ValueError('invalid option {}'.format(virus))

    anchor = None
    for idx, record in enumerate(SeqIO.parse(train_fname, 'fasta')):
        if record.id == anchor_id:
            anchor = str(record.seq)
    assert(anchor is not None)

    # read embedding file
    embeddings = {}
    with open(embed_fname) as f:
        for line in f:
            if line.startswith('>'):
                name = line.rstrip()[1:]
            embedding = np.array(
                [ float(field)
                  for field in f.readline().rstrip().split() ]
            )
            embeddings[name] = embedding
    base_embedding = embeddings['base']

    mutations = [
        str(record.seq) for record in SeqIO.parse(mut_fname, 'fasta')
    ]
    mut2change = {}
    for mutation in mutations:
        didx = [ c1 != c2
                 for c1, c2 in zip(anchor, mutation) ].index(True) # position of first difference
        embedding = embeddings['mut_{}_{}'.format(didx, mutation[didx])]
        mutation_clean = mutation.replace('-', '')
        mut2change[mutation_clean] = abs(base_embedding - embedding).sum()

    anchor = anchor.replace('-', '')
    escape_idx, changes = [], []
    for i in range(len(anchor)):
        if virus == 'bg505' and (i < 29 or i > 698):
            continue
        if virus == 'cov2rbd' and (i < 318 or i > 540):
            continue
        for word in vocabulary:
            if anchor[i] == word:
                continue # skip mutations without an effect
            mut_seq = anchor[:i] + word + anchor[i + 1:] # perform mutations
            if mut_seq in seqs_escape and \
               (sum([ m['significant'] for m in seqs_escape[mut_seq] ]) > 0):
                escape_idx.append(len(changes))
            changes.append(mut2change[mut_seq])
    changes = np.array(changes) # np format for data analysis

    plot_result(-changes, escape_idx, virus, 'bepler', legend_name='Bepler')

if __name__ == '__main__':
    """
    Performs escape prediction benchmarking using the model specified in the method CLI argument.
    
    Accepted methods: 'bepler', 'energy', 'evcouplings', 'freq', 'tape', 'unirep'
    """
    args = parse_args()

    vocabulary = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V', 'U',
    ]

    if args.method == 'bepler':
        escape_bepler(args.virus, vocabulary)

    elif args.method == 'energy':
        escape_energy(args.virus, vocabulary)

    elif args.method == 'evcouplings':
        escape_evcouplings(args.virus, vocabulary)

    elif args.method == 'freq':
        escape_freq(args.virus, vocabulary)

    elif args.method == 'tape':
        escape_tape(args.virus, vocabulary)

    elif args.method == 'unirep':
        escape_tape(args.virus, vocabulary, 'unirep')
