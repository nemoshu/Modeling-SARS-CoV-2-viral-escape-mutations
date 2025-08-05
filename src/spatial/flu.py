from mutation import *

np.random.seed(1)
random.seed(1)

def parse_args():
    """
    Configures analysis pipeline from CLI.

     Returns:
        argparse.Namespace
    """
    import argparse
    parser = argparse.ArgumentParser(description='Flu sequence analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., hmm, lstm)')
    parser.add_argument('--namespace', type=str, default='influenza',
                        help='Model namespace')
    parser.add_argument('--dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Training minibatch size')
    parser.add_argument('--n-epochs', type=int, default=14,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint')
    parser.add_argument('--train', action='store_true',
                        help='Train model')
    parser.add_argument('--train-split', action='store_true',
                        help='Train model on portion of data')
    parser.add_argument('--test', action='store_true',
                        help='Test model')
    parser.add_argument('--embed', action='store_true',
                        help='Analyze embeddings')
    parser.add_argument('--semantics', action='store_true',
                        help='Analyze mutational semantic change')
    parser.add_argument('--combfit', action='store_true',
                        help='Analyze combinatorial fitness')
    args = parser.parse_args()
    return args

def load_meta(meta_fnames):
    """
    Loads metadata from files.

    Args:
        meta_fnames (list[str]): The list of file names containing metadata

    Returns:
        metas (dict[str, list[dict]]): The metadata dictionary
    """
    metas = {}
    for fname in meta_fnames:
        with open(fname) as f:
            header = f.readline().rstrip().split('\t')
            for line in f:
                fields = line.rstrip().split('\t')
                accession = fields[1]
                meta = {}
                for key, value in zip(header, fields):
                    if key == 'Subtype':
                        meta[key] = value.strip('()').split('N')[0].split('/')[-1]
                    elif key == 'Collection Date':
                        meta[key] = int(value.split('/')[-1]) \
                                    if value != '-N/A-' else None
                    elif key == 'Host Species':
                        meta[key] = value.split(':')[1].split('/')[-1].lower()
                    else:
                        meta[key] = value
                metas[accession] = meta
    return metas

def process(fnames, meta_fnames):
    """
    Process metadata to parse and filter sequences.

    Args:
        fnames (list[str]): The list of filenames of FASTA files.
        meta_fnames (list[str]): The list of filenames of Metadata TSVs.

    Returns:
        seqs (dict[str, list[dict]]): dictionary mapping unique sequences to lists of corresponding metadata
    """
    metas = load_meta(meta_fnames)

    seqs = {}
    for fname in fnames:
        for record in SeqIO.parse(fname, 'fasta'):
            if 'Reference_Perth2009_HA_coding_sequence' in record.description:
                continue # skip header
            if str(record.seq).count('X') > 10:
                continue # skip if there are more than 10 ambiguous records
            if record.seq not in seqs:
                seqs[record.seq] = []
            accession = record.description.split('|')[0].split(':')[1]
            seqs[record.seq].append(metas[accession])
    return seqs

def split_seqs(seqs):
    """
    Splitting sequences into training and test sets.

    Args:
        seqs (dict[str, list[dict]]): The sequences dictionary

    Returns:
        tuple:
            train_seqs (dict[str, list[dict]]): The training sequences
            test_seqs (dict[str, list[dict]]): The test sequences
    """
    train_seqs, test_seqs, val_seqs = {}, {}, {}

    old_cutoff = 1990
    new_cutoff = 2018

    tprint('Splitting seqs...')
    for seq in seqs:
        # Pick validation set based on date.
        seq_dates = [
            meta['Collection Date'] for meta in seqs[seq]
            if meta['Collection Date'] is not None
        ] # dates of the sequence list
        if len(seq_dates) > 0:
            oldest_date = sorted(seq_dates)[0]
            if oldest_date < old_cutoff or oldest_date >= new_cutoff:
                test_seqs[seq] = seqs[seq]
                continue # if date is outside the range, add to test sequences list instead
        train_seqs[seq] = seqs[seq]
    tprint('{} train seqs, {} test seqs.'
           .format(len(train_seqs), len(test_seqs)))

    return train_seqs, test_seqs

def setup(args):
    """
    Constructs the model and loads the influenza sequence data.

    Args:
        args (argparse.Namespace): The arguments parsed from the command line

    Returns:
        Tuple:
            model (object): The model object
            seqs (dict[str, list[dict]]): The sequences dictionary
    """

    fnames = [ 'data/influenza/ird_influenzaA_HA_allspecies.fa' ]
    meta_fnames = [ 'data/influenza/ird_influenzaA_HA_allspecies_meta.tsv' ]

    seqs = process(fnames, meta_fnames)

    seq_len = max([ len(seq) for seq in seqs ]) + 2
    vocab_size = len(AAs) + 2

    model = get_model(args, seq_len, vocab_size)

    return model, seqs

def interpret_clusters(adata):
    """
    Interprets and prints the contents of each Louvain cluster.

    Args:
        adata (anndata.AnnData): Annotated data object

    Returns:
        None
    """
    clusters = sorted(set(adata.obs['louvain']))
    for cluster in clusters:
        tprint('Cluster {}'.format(cluster))
        adata_cluster = adata[adata.obs['louvain'] == cluster]
        for var in [ 'Collection Date', 'Country', 'Subtype',
                     'Flu Season', 'Host Species', 'Strain Name' ]:
            tprint('\t{}:'.format(var))
            counts = Counter(adata_cluster.obs[var]) # information regarding the most common cluster
            for val, count in counts.most_common():
                tprint('\t\t{}: {}'.format(val, count))
        tprint('')

    # construct cluster-to-subtype and cluster-to-species maps
    cluster2subtype = {}
    cluster2species = {}
    for i in range(len(adata)):
        cluster = adata.obs['louvain'][i]
        if cluster not in cluster2subtype:
            cluster2subtype[cluster] = []
            cluster2species[cluster] = []
        cluster2subtype[cluster].append(adata.obs['Subtype'][i])
        cluster2species[cluster].append(adata.obs['Host Species'][i])

    # finds most frequent subtype and species for each cluster
    # and calculates the relevant percentages
    largest_pct_subtype = []
    largest_pct_species = []
    for cluster in cluster2subtype:
        count = Counter(cluster2subtype[cluster]).most_common(1)[0][1]
        largest_pct_subtype.append(float(count) /
                                   len(cluster2subtype[cluster]))
        count = Counter(cluster2species[cluster]).most_common(1)[0][1]
        largest_pct_species.append(float(count) /
                                   len(cluster2species[cluster]))


    for idx, pct in enumerate(largest_pct_subtype):
        tprint('\tCluster {}, largest subtype % = {}'.format(idx, pct))
    for idx, pct in enumerate(largest_pct_species):
        tprint('\tCluster {}, largest species % = {}'.format(idx, pct))

    tprint('Purity, Louvain and subtype: {}'
           .format(np.mean(largest_pct_subtype)))
    tprint('Purity, Louvain and host species: {}'
           .format(np.mean(largest_pct_species)))

def seq_clusters(adata):
    """
    Writes most common sequences to FASTA.

    Args:
        adata (anndata.AnnData): Annotated data object

    Returns:
        None
    """
    clusters = sorted(set(adata.obs['louvain']))
    for cluster in clusters:
        adata_cluster = adata[adata.obs['louvain'] == cluster]
        counts = Counter(adata_cluster.obs['seq'])
        with open('target/influenza/clusters/cluster{}.fa'.format(cluster), 'w') as of:
            for i, (seq, count) in enumerate(counts.most_common()):
                of.write('>cluster{}_{}_{}\n'.format(cluster, i, count))
                of.write(seq + '\n\n')

def plot_umap(adata, namespace='influenza'):
    """
    Generates and saves UMAP visualizations.

    Args:
        adata (anndata.AnnData): Annotated data object
        namespace (str): Prefix namespace for output filenames. Defaults to 'influenza'.

    Returns:
        None
    """
    if namespace == 'flu1918':
        plt.figure()
        ax = plt.gca()
        sc.pl.umap(adata, color='Host Species', ax=ax, size=20)
        ratio = 0.3
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
        plt.savefig('figures/umap_{}_species.png'.format(namespace))
        plt.close()
    else:
        sc.pl.umap(adata, color='Host Species',
                   save='_{}_species.png'.format(namespace))

    sc.pl.umap(adata, color='Subtype',
               save='_{}_subtype.png'.format(namespace))
    sc.pl.umap(adata, color='Collection Date',
               save='_{}_date.png'.format(namespace))
    sc.pl.umap(adata, color='louvain',
               save='_{}_louvain.png'.format(namespace))

def analyze_embedding(args, model, seqs, vocabulary):
    """
    Embeds sequences, analyses them via clustering, and visualizes them.

    Args:
        args (argparse.Namespace): Arguments object from argparse
        model (object): The model object
        seqs (dict[str, list[dict]]): The sequences dictionary
        vocabulary (dict[str, int]): the dictionary of the AA vocabulary
    Returns:
        None
    """
    seqs = embed_seqs(args, model, seqs, vocabulary, use_cache=True)

    X, obs = [], {}
    obs['n_seq'] = [] # number of times each sequence appears
    obs['seq'] = [] # string form of each sequence
    for seq in seqs:
        meta = seqs[seq][0]
        X.append(meta['embedding'].mean(0))
        for key in meta:
            if key == 'embedding':
                continue # skip header
            if key not in obs:
                obs[key] = []
            obs[key].append(Counter([
                meta[key] for meta in seqs[seq]
            ]).most_common(1)[0][0])
        obs['n_seq'].append(len(seqs[seq]))
        obs['seq'].append(str(seq))
    X = np.array(X)

    adata = AnnData(X)
    for key in obs:
        adata.obs[key] = obs[key]
    adata = adata[
        np.logical_or.reduce((
            adata.obs['Host Species'] == 'human',
            adata.obs['Host Species'] == 'avian',
            adata.obs['Host Species'] == 'swine',
        ))
    ]

    sc.pp.neighbors(adata, n_neighbors=100, use_rep='X')
    sc.tl.louvain(adata, resolution=1.)

    sc.set_figure_params(dpi_save=500)

    sc.tl.umap(adata, min_dist=1.)
    plot_umap(adata)
    plot_umap(adata[adata.obs['louvain'] == '30'],
              namespace='flu1918')

    # output and save clusters
    interpret_clusters(adata)

    seq_clusters(adata)

if __name__ == '__main__':
    """
    Runs training for Influenza datasets as specified by the command line arguments.
    """
    args = parse_args()

    AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V', 'X', 'Z', 'J', 'U', 'B', 'Z'
    ] # available amino acids

    vocabulary = { aa: idx + 1 for idx, aa in enumerate(sorted(AAs)) } # vocabulary - amino acid to index map

    model, seqs = setup(args)

    if args.checkpoint is not None:
        model.model_.load_weights(args.checkpoint)
        tprint('Model summary:')
        tprint(model.model_.summary())

    if args.train or args.train_split or args.test:
        train_test(args, model, seqs, vocabulary, split_seqs)

    if args.embed:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')
        no_embed = { 'hmm' }
        if args.model_name in no_embed:
            raise ValueError('Embeddings not available for models: {}'
                             .format(', '.join(no_embed)))
        analyze_embedding(args, model, seqs, vocabulary)

    if args.semantics:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')

        from escape import load_doud2018, load_lee2019

        tprint('Lee et al. 2018...')
        seq_to_mutate, escape_seqs = load_doud2018()
        analyze_semantics(args, model, vocabulary, seq_to_mutate, escape_seqs,
                          prob_cutoff=0., beta=1., plot_acquisition=True,
                          plot_namespace='flu_h1')
        tprint('')

        tprint('Lee et al. 2019...')
        seq_to_mutate, escape_seqs = load_lee2019()
        analyze_semantics(args, model, vocabulary, seq_to_mutate, escape_seqs,
                          prob_cutoff=0., beta=1., plot_acquisition=True,
                          plot_namespace='flu_h3')

    if args.combfit:
        from combinatorial_fitness import load_doud2016
        tprint('Doud et al. 2016...')
        wt_seqs, seqs_fitness = load_doud2016()
        strains = sorted(wt_seqs.keys())
        for strain in strains:
            analyze_comb_fitness(args, model, vocabulary,
                                 strain, wt_seqs[strain], seqs_fitness,
                                 prob_cutoff=0., beta=1.)

        from combinatorial_fitness import load_wu2020
        tprint('Wu et al. 2020...')
        wt_seqs, seqs_fitness = load_wu2020()
        strains = sorted(wt_seqs.keys())
        for strain in strains:
            analyze_comb_fitness(args, model, vocabulary,
                                 strain, wt_seqs[strain], seqs_fitness,
                                 prob_cutoff=0., beta=1.)
