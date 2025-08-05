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
    parser = argparse.ArgumentParser(description='Coronavirus sequence analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., hmm, lstm)')
    parser.add_argument('--namespace', type=str, default='cov',
                        help='Model namespace')
    parser.add_argument('--dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Training minibatch size')
    parser.add_argument('--n-epochs', type=int, default=11,
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
    parser.add_argument('--reinfection', action='store_true',
                        help='Analyze reinfection cases')
    args = parser.parse_args()
    return args

def parse_viprbrc(entry):
    """
    Parses the VIPRBRC dataset to metadata.

    Args:
        entry (str): header string for viprbrc entry

    Returns:
        meta (dict): metadata of the viprbrc entry
    """
    fields = entry.split('|')
    if fields[7] == 'NA':
        date = None
    else:
        date = fields[7].split('/')[0]
        date = dparse(date.replace('_', '-'))

    country = fields[9]
    from locations import country2continent
    if country in country2continent:
        continent = country2continent[country]
    else:
        country = 'NA'
        continent = 'NA'

    from mammals import species2group

    meta = {
        'strain': fields[5],
        'host': fields[8],
        'group': species2group[fields[8]],
        'country': country,
        'continent': continent,
        'dataset': 'viprbrc',
    }
    return meta

def parse_nih(entry):
    """
    Parses the NIH dataset to metadata.
    Args:
        entry (str): header string for nih entry
    Returns:
        meta (dict): metadata of the nih entry
    """
    fields = entry.split('|')

    country = fields[3]
    from locations import country2continent
    if country in country2continent:
        continent = country2continent[country]
    else:
        country = 'NA'
        continent = 'NA'

    meta = {
        'strain': 'SARS-CoV-2',
        'host': 'human',
        'group': 'human',
        'country': country,
        'continent': continent,
        'dataset': 'nih',
    }
    return meta

def parse_gisaid(entry):
    """
    Parses the GISAID dataset to metadata.

    Args:
        entry (str): header string for gisaid entry
    Returns:
        meta (dict): metadata of the gisaid entry
    """
    fields = entry.split('|')

    type_id = fields[1].split('/')[1]

    if type_id in { 'bat', 'canine', 'cat', 'env', 'mink',
                    'pangolin', 'tiger' }:
        host = type_id
        country = 'NA'
        continent = 'NA'
    else:
        host = 'human'
        from locations import country2continent
        if type_id in country2continent:
            country = type_id
            continent = country2continent[country]
        else:
            country = 'NA'
            continent = 'NA'

    from mammals import species2group

    meta = {
        'strain': fields[1],
        'host': host,
        'group': species2group[host].lower(),
        'country': country,
        'continent': continent,
        'dataset': 'gisaid',
    }
    return meta

def process(fnames):
    """
    Processes metadata, filters sequences such that they contain at least 1000 AA and no ambiguous residues,
    and writes consolidated metadata.

    Args:
        fnames (list[str]): the list FASTA file names
    Returns:
        seqs(dict[str, list[metadata]]): the dictionary of the processed sequences
    """
    seqs = {}
    for fname in fnames:
        for record in SeqIO.parse(fname, 'fasta'):
            if len(record.seq) < 1000:
                continue # if sequence is too short, skip
            if str(record.seq).count('X') > 0:
                continue # if there is an ambiguous residue ('X'), skip
            if record.seq not in seqs:
                seqs[record.seq] = []
            if fname == 'data/cov/viprbrc_db.fasta':
                meta = parse_viprbrc(record.description)
            elif fname == 'data/cov/gisaid.fasta':
                meta = parse_gisaid(record.description)
            else:
                meta = parse_nih(record.description)
            meta['accession'] = record.description
            seqs[record.seq].append(meta)

    # writes filtered records to cov_all.fa
    with open('data/cov/cov_all.fa', 'w') as of:
        for seq in seqs:
            metas = seqs[seq]
            for meta in metas:
                of.write('>{}\n'.format(meta['accession']))
                of.write('{}\n'.format(str(seq)))
    return seqs

def split_seqs(seqs):
    """
    Splits sequences into training and test sets.

    Args:
        seqs(dict[str, list[metadata]]): the dictionary of the processed sequences
    Returns:
        train_seqs (dict[str, list[metadata]]): the training sequences
        test_seqs (dict[str, list[metadata]]): the test sequences
    """
    train_seqs, test_seqs = {}, {}

    tprint('Splitting seqs...')
    for idx, seq in enumerate(seqs):
        if idx % 10 < 2:
            test_seqs[seq] = seqs[seq]
        else:
            train_seqs[seq] = seqs[seq]
    tprint('{} train seqs, {} test seqs.'
           .format(len(train_seqs), len(test_seqs)))

    return train_seqs, test_seqs

def setup(args):
    """
    Sets up the analysis pipeline.

    Args:
        args (argparse.Namespace): command line arguments

    Returns:
        model (torch.nn.Module): untrained neural network model to be used
        seqs(dict[str, list[metadata]]): the dictionary of the processed sequences

    Input Files:
        - data/cov/sars_cov2_seqs.fa
        - data/cov/viprbrc_db.fasta
        - data/cov/gisaid.fasta
    """
    fnames = [ 'data/cov/sars_cov2_seqs.fa',
               'data/cov/viprbrc_db.fasta',
               'data/cov/gisaid.fasta' ]

    seqs = process(fnames)

    seq_len = max([ len(seq) for seq in seqs ]) + 2
    vocab_size = len(AAs) + 2

    model = get_model(args, seq_len, vocab_size,
                      inference_batch_size=1200)

    return model, seqs

def interpret_clusters(adata):
    """
    Prints the most common values of host, country and strain in adata.

    Args:
        adata (anndata.AnnData): Annotated data object

    Returns:
        None
    """
    clusters = sorted(set(adata.obs['louvain']))
    for cluster in clusters:
        tprint('Cluster {}'.format(cluster))
        adata_cluster = adata[adata.obs['louvain'] == cluster]
        for var in [ 'host', 'country', 'strain' ]:
            tprint('\t{}:'.format(var))
            counts = Counter(adata_cluster.obs[var])
            for val, count in counts.most_common():
                tprint('\t\t{}: {}'.format(val, count))
        tprint('')

def plot_umap(adata, categories, namespace='cov'):
    """
    Generates and saves UMAP plots for each category.

    Args:
        adata (anndata.AnnData): Annotated data object
        categories (list[str]): Metadata columns to use for coloring
        namespace (str): Prefix namespace for output filenames. Defaults to 'cov'.

    Output File:
        - figures/{namespace}umap{category}.png
    """
    for category in categories:
        sc.pl.umap(adata, color=category,
                   save='_{}_{}.png'.format(namespace, category))

def analyze_embedding(args, model, seqs, vocabulary):
    """
    Generates sequence embeddings, and visualizes clusters using UMAP and Louvain clusters.

    Also generates and saves annotated UMAP plots.

    Args:
        args (argparse.Namespace): command line arguments
        model (torch.nn.Module): the neural network model to be used
        seqs (dict[str, list]): the dictionary of the sequences
        vocabulary (dict[str, int]): the dictionary of the AA vocabulary

    Returns:
        None
    """

    seqs = embed_seqs(args, model, seqs, vocabulary, use_cache=True) # embed sequences

    X, obs = [], {}
    obs['n_seq'] = [] # number of times each sequence appears
    obs['seq'] = [] # string form of each sequence
    for seq in seqs:
        meta = seqs[seq][0]
        X.append(meta['embedding'].mean(0)) # average embedding vectors per sequence
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

    sc.pp.neighbors(adata, n_neighbors=20, use_rep='X')
    sc.tl.louvain(adata, resolution=1.)
    sc.tl.umap(adata, min_dist=1.)

    sc.set_figure_params(dpi_save=500)
    plot_umap(adata, [ 'host', 'group', 'continent', 'louvain' ])

    interpret_clusters(adata)

    adata_cov2 = adata[(adata.obs['louvain'] == '0') |
                       (adata.obs['louvain'] == '2')]
    plot_umap(adata_cov2, [ 'host', 'group', 'country' ],
              namespace='cov7') # saves coloured maps

if __name__ == '__main__':
    """
    Runs training for SARS-COV-2 datasets as specified by the command line arguments.
    """
    args = parse_args()

    AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V', 'X', 'Z', 'J', 'U', 'B',
    ] # available amino acids

    vocabulary = { aa: idx + 1 for idx, aa in enumerate(sorted(AAs)) } # vocabulary - amino acid to index map

    model, seqs = setup(args)

    if args.checkpoint is not None:
        model.model_.load_weights(args.checkpoint)
        tprint('Model summary:')
        tprint(model.model_.summary())

    if args.train:
        batch_train(args, model, seqs, vocabulary, batch_size=1000)

    if args.train_split or args.test:
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

        from escape import load_baum2020, load_greaney2020
        tprint('Baum et al. 2020...')
        seq_to_mutate, seqs_escape = load_baum2020()
        analyze_semantics(args, model, vocabulary,
                          seq_to_mutate, seqs_escape, comb_batch=10000,
                          prob_cutoff=0, beta=1., plot_acquisition=True,)
        tprint('Greaney et al. 2020...')
        seq_to_mutate, seqs_escape = load_greaney2020()
        analyze_semantics(args, model, vocabulary,
                          seq_to_mutate, seqs_escape, comb_batch=10000,
                          min_pos=318, max_pos=540, # Restrict to RBD.
                          prob_cutoff=0, beta=1., plot_acquisition=True,
                          plot_namespace='cov2rbd')

    if args.combfit:
        from combinatorial_fitness import load_starr2020
        tprint('Starr et al. 2020...')
        wt_seqs, seqs_fitness = load_starr2020()
        strains = sorted(wt_seqs.keys())
        for strain in strains:
            analyze_comb_fitness(args, model, vocabulary,
                                 strain, wt_seqs[strain], seqs_fitness,
                                 comb_batch=10000, prob_cutoff=0., beta=1.)

    if args.reinfection:
        from reinfection import load_to2020, load_ratg13, load_sarscov1
        from plot_reinfection import plot_reinfection

        tprint('To et al. 2020...')
        wt_seq, mutants = load_to2020()
        analyze_reinfection(args, model, seqs, vocabulary, wt_seq, mutants,
                            namespace='to2020')
        plot_reinfection(namespace='to2020')
        null_combinatorial_fitness(args, model, seqs, vocabulary,
                                   wt_seq, mutants, n_permutations=100000000,
                                   namespace='to2020')

        tprint('Positive controls...')
        wt_seq, mutants = load_ratg13()
        analyze_reinfection(args, model, seqs, vocabulary, wt_seq, mutants,
                            namespace='ratg13')
        plot_reinfection(namespace='ratg13')
        wt_seq, mutants = load_sarscov1()
        analyze_reinfection(args, model, seqs, vocabulary, wt_seq, mutants,
                            namespace='sarscov1')
        plot_reinfection(namespace='sarscov1')
