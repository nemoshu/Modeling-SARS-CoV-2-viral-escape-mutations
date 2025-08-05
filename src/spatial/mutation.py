from utils import *

def err_model(name):
    raise ValueError('Model {} not supported'.format(name))

def get_model(args, seq_len, vocab_size,
              inference_batch_size=1500):
    """
    Obtains the relevant untrained model specified in the command line arguments.

    Args:
        args (argparse.Namespace): Arguments parsed from the command line.
        seq_len (int): Sequence length.
        vocab_size (int): Vocabulary size.
        inference_batch_size (int): Batch size. Defaults to 1500.

    Returns:
        An untrained model of the specified type

    Raises:
        ValueError: If the specified model name is not supported.
    """
    if args.model_name == 'hmm':
        from hmmlearn.hmm import MultinomialHMM
        model = MultinomialHMM(
            n_components=args.dim,
            startprob_prior=1.0,
            transmat_prior=1.0,
            algorithm='viterbi',
            random_state=1,
            tol=0.01,
            verbose=True,
            params='ste',
            init_params='ste'
        )
    elif args.model_name == 'dnn':
        from language_model import DNNLanguageModel
        model = DNNLanguageModel(
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=args.dim,
            n_hidden=2,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            inference_batch_size=inference_batch_size,
            cache_dir='target/{}'.format(args.namespace),
            seed=args.seed,
            verbose=True,
        )
    elif args.model_name == 'lstm':
        from language_model import LSTMLanguageModel
        model = LSTMLanguageModel(
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=args.dim,
            n_hidden=2,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            inference_batch_size=inference_batch_size,
            cache_dir='target/{}'.format(args.namespace),
            seed=args.seed,
            verbose=True,
        )
    elif args.model_name == 'bilstm':
        from language_model import BiLSTMLanguageModel
        model = BiLSTMLanguageModel(
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=args.dim,
            n_hidden=2,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            inference_batch_size=inference_batch_size,
            cache_dir='target/{}'.format(args.namespace),
            seed=args.seed,
            verbose=True,
        )
    elif args.model_name == 'attention':
        from language_model import AttentionLanguageModel
        model = AttentionLanguageModel(
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=args.dim,
            n_hidden=2,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            inference_batch_size=inference_batch_size,
            cache_dir='target/{}'.format(args.namespace),
            seed=args.seed,
            verbose=True,
        )
    else:
        err_model(args.model_name)

    return model

def featurize_seqs(seqs, vocabulary):
    """
    Converts a dictionary of sequences into arrays used by the models.

    Args:
        seqs (dict): dictionary of sequences
        vocabulary (dict[str, int]): dictionary of residues to a unique index

    Returns:
        X (np.ndarray): concatenated token indices (including start/end tokens)
        lens (np.ndarray): length of each sequence (including start/end tokens)
    """
    start_int = len(vocabulary) + 1
    end_int = len(vocabulary) + 2
    sorted_seqs = sorted(seqs.keys())
    X = np.concatenate([
        np.array([ start_int ] + [
            vocabulary[word] for word in seq
        ] + [ end_int ]) for seq in sorted_seqs
    ]).reshape(-1, 1)
    lens = np.array([ len(seq) + 2 for seq in sorted_seqs ])
    assert(sum(lens) == X.shape[0]) # confirm that total length is the same as the expected inpit size
    return X, lens

def fit_model(name, model, seqs, vocabulary):
    """
    Vectorizes data and calls model.fit() to fit the model.

    Args:
        name (str): unused
        model (LanguageModel): A model which implements .fit(X, lengths)
        seqs (dict): dictionary of sequences
        vocabulary (dict[str, int]): dictionary of residues to a unique index
    Returns:
        model (LanguageModel): The model trained on seqs
    """
    X, lengths = featurize_seqs(seqs, vocabulary)
    model.fit(X, lengths)
    return model

def cross_entropy(logprob, n_samples):
    """
    Converts total log-likelihood (sum of log-probabilities) into per-sample cross-entropy loss.

    Args:
        logprob (float): sum of log-likelihoods from model.score
        n_samples (int): total number of samples

    Returns:
        float: cross-entropy loss
    """
    return -logprob / n_samples

def report_performance(model_name, model, vocabulary,
                       train_seqs, test_seqs):
    """
    Evaluates and prints cross-entropy performance.

    Args:
        model_name (str): name of the model
        model (LanguageModel): the model instance
        vocabulary (dict[str, int]): dictionary of residues to a unique index
        train_seqs (dict): dictionary of train sequences
        test_seqs (dict): dictionary of test sequences
    """
    X_train, lengths_train = featurize_seqs(train_seqs, vocabulary)
    logprob = model.score(X_train, lengths_train)
    tprint('Model {}, train cross entropy: {}'
           .format(model_name, cross_entropy(logprob, len(lengths_train))))
    X_test, lengths_test = featurize_seqs(test_seqs, vocabulary)
    logprob = model.score(X_test, lengths_test)
    tprint('Model {}, test cross entropy: {}'
           .format(model_name, cross_entropy(logprob, len(lengths_test))))

def train_test(args, model, seqs, vocabulary, split_seqs=None):
    """
    Handles one round of training.

    Args:
        args (argparse.Namespace): command line arguments
        model (LanguageModel): the model instance
        seqs (dict): dictionary of sequences
        vocabulary (dict[str, int]): dictionary of residues to a unique index
        split_seqs: callable that splits a dictionary of sequences into training and test sequences

    Returns:
        Trained model, if args.train is True

    Raises:
        ValueError: if both args.train and args.train_split are true, or if split_seqs is not a callable
    """
    if args.train and args.train_split:
        raise ValueError('Training on full and split data is invalid.')

    if args.train:
        model = fit_model(args.model_name, model, seqs, vocabulary)
        return

    if split_seqs is None:
        raise ValueError('Must provide function to split train/test.')
    train_seqs, val_seqs = split_seqs(seqs)

    if args.train_split:
        model = fit_model(args.model_name, model, train_seqs, vocabulary)
    if args.test:
        report_performance(args.model_name, model, vocabulary,
                           train_seqs, val_seqs)

def batch_train(args, model, seqs, vocabulary, batch_size=5000,
                verbose=True):
    """
    Trains mini-batches of sequences, and saves checkpoints

    Args:
        args (argparse.Namespace): command line arguments
        model (LanguageModel): the model instance
        seqs (dict): dictionary of sequences
        vocabulary (dict[str, int]): dictionary of residues to a unique index
        batch_size (int): number of sequences per batch, defaults to 5000
        verbose (bool): whether to use verbose settings, defaults to True

    Raises:
        AssertionError: if args.train is False
    """
    assert(args.train)

    # Control epochs here.
    n_epochs = args.n_epochs
    args.n_epochs = 1
    model.n_epochs_ = 1

    n_batches = math.ceil(len(seqs) / float(batch_size))
    if verbose:
        tprint('Train seq batch size: {}, N batches: {}'
               .format(batch_size, n_batches))

    for epoch in range(n_epochs):
        if verbose:
            tprint('True epoch {}/{}'.format(epoch + 1, n_epochs))
        perm_seqs = [ str(s) for s in seqs.keys() ]
        random.shuffle(perm_seqs)

        # split into batches, and train model for each batch
        for batchi in range(n_batches):
            start = batchi * batch_size
            end = (batchi + 1) * batch_size
            seqs_batch = { seq: seqs[seq] for seq in perm_seqs[start:end] }
            train_test(args, model, seqs_batch, vocabulary)
            del seqs_batch

        fname_prefix = ('target/{0}/checkpoints/{1}/{1}_{2}'
                        .format(args.namespace, args.model_name, args.dim))

        if epoch == 0:
            os.rename('{}-01.hdf5'.format(fname_prefix),
                      '{}-00.hdf5'.format(fname_prefix))
        else:
            os.rename('{}-01.hdf5'.format(fname_prefix),
                      '{}-{:02d}.hdf5'.format(fname_prefix, epoch + 1))
    os.rename('{}-00.hdf5'.format(fname_prefix),
              '{}-01.hdf5'.format(fname_prefix))

def embed_seqs(args, model, seqs, vocabulary,
               use_cache=False, verbose=True):
    """
    Computes semantic embeddings for each sequence.

    Args:
        args (argparse.Namespace): command line arguments
        model (LanguageModel): the model instance
        seqs (dict): dictionary of sequences
        vocabulary (dict[str, int]): dictionary of residues to a unique index
        use_cache (bool): whether to load/save .npy files
    Returns:
        The original seqs dict with new key 'embedding' in each metadata indicating model's embedding dimension
    """
    X_cat, lengths = featurize_seqs(seqs, vocabulary)

    if use_cache:
        mkdir_p('target/{}/embedding'.format(args.namespace))
        embed_fname = ('target/{}/embedding/{}_{}.npy'
                       .format(args.namespace, args.model_name, args.dim))
    else:
        embed_fname = None

    if use_cache and os.path.exists(embed_fname):
        X_embed = np.load(embed_fname, allow_pickle=True) # use embedding cache if enabled and exists
    else:
        X_embed = model.transform(X_cat, lengths, embed_fname)
        if use_cache:
            np.save(embed_fname, X_embed)

    sorted_seqs = sorted(seqs)
    for seq_idx, seq in enumerate(sorted_seqs):
        for meta in seqs[seq]:
            meta['embedding'] = X_embed[seq_idx] # add embedding to metadata

    return seqs

def predict_sequence_prob(args, seq_of_interest, vocabulary, model,
                          verbose=False):
    """
    Predict per-position grammatical probabilities from the trained model.

    Args:
        args (argparse.Namespace): command line arguments
        seq_of_interest (str): single sequence to predict
        vocabulary (dict[str, int]): dictionary of residues to a unique index
        model (LanguageModel): the model instance
        verbose (bool): whether to use verbose settings, defaults to False

    Returns:
        y_pred (np.ndarray): each row is the distribution over the vocabulary at that position (includes start/end tokens)
    """
    seqs = { seq_of_interest: [ {} ] }
    X_cat, lengths = featurize_seqs(seqs, vocabulary)

    y_pred = model.predict(X_cat, lengths)
    assert(y_pred.shape[0] == len(seq_of_interest) + 2) # confirm that the prediction is of the expected shape

    return y_pred

def analyze_comb_fitness(
        args, model, vocabulary, strain, wt_seq, seqs_fitness,
        comb_batch=None, prob_cutoff=0., beta=1., verbose=True,
):
    """
    Analyses combinatorial fitness for multi-mutation sequences, and generates combinatorial fitness scatter plots

    Args:
        args (argparse.Namespace): command line arguments
        model (LanguageModel): the model instance
        vocabulary (dict[string, int]): dictionary of residues to a unique index
        strain (string): name of the strain
        wt_seq (string): sequence to analyse
        seqs_fitness (dict[(string, string), list[dict]]): the fitness of sequences - maps (mutant_seq, strain) to a list of metadata dicts
        comb_batch (int): how many mutants to embed per batch
        prob_cutoff (float): the minimum per-mutation probability
        beta (float): beta parameter
        verbose (bool): whether to use verbose settings, defaults to True
    """
    from copy import deepcopy
    seqs_fitness = { seq: seqs_fitness[(seq, strain_i)]
                     for seq, strain_i in seqs_fitness
                     if strain_i == strain }

    y_pred = predict_sequence_prob(
        args, wt_seq, vocabulary, model, verbose=verbose
    )

    # construct word position probability map from prediction vector
    word_pos_prob = {}
    for pos in range(len(wt_seq)):
        for word in vocabulary:
            word_idx = vocabulary[word]
            prob = y_pred[pos + 1, word_idx]
            if prob < prob_cutoff:
                continue
            word_pos_prob[(word, pos)] = prob

    # obtain embedding
    base_embedding = embed_seqs(
        args, model, { wt_seq: [ {} ] }, vocabulary,
        use_cache=False, verbose=False
    )[wt_seq][0]['embedding']

    if comb_batch is None:
        comb_batch = len(seqs_fitness)
    seqs = sorted(seqs_fitness.keys())
    n_batches = math.ceil(float(len(seqs)) / comb_batch)

    data = []
    for batchi in range(n_batches):
        # split into batches
        start = batchi * comb_batch
        end = (batchi + 1) * comb_batch
        seqs_fitness_batch = {
            seq: deepcopy(seqs_fitness[seq])
            for seq in seqs[start:end]
            if seqs_fitness[seq][0]['strain'] == strain
        }

        # embed
        seqs_fitness_batch = embed_seqs(
            args, model, seqs_fitness_batch, vocabulary,
            use_cache=False, verbose=False
        )

        for mut_seq in seqs_fitness_batch:
            assert(len(seqs_fitness_batch[mut_seq]) == 1) # confirm only one entry exists
            meta = seqs_fitness_batch[mut_seq][0]
            if meta['strain'] != strain:
                continue
            assert(len(mut_seq) == len(wt_seq)) # length of mutation sequence should match length of wild type sequence

            # obtain mutation and raw probabilities
            mut_pos = set(meta['mut_pos'])
            raw_probs = []
            for idx, aa in enumerate(mut_seq):
                if idx in mut_pos:
                    raw_probs.append(word_pos_prob[(aa, idx)])
                else:
                    assert(aa == wt_seq[idx])
            assert(len(raw_probs) == len(mut_pos))

            # grammar fitness and semantic change
            grammar = sum(np.log10(raw_probs))
            sem_change = abs(base_embedding - meta['embedding']).sum()

            # add to dataset
            data.append([
                meta['strain'],
                meta['fitness'],
                meta['preference'],
                grammar, # grammar fitness
                sem_change, # semantic change
                sem_change + (beta * grammar), # cscs
            ])

        del seqs_fitness_batch

    df = pd.DataFrame(data, columns=[
        'strain', 'fitness', 'preference',
        'predicted', 'sem_change', 'cscs'
    ])

    # report data and plot graphs
    print('\nStrain: {}'.format(strain))
    print('\tGrammaticality correlation:')
    print('\t\tSpearman r = {:.4f}, P = {:.4g}'
          .format(*ss.spearmanr(df.preference, df.predicted)))
    print('\t\tPearson rho = {:.4f}, P = {:.4g}'
          .format(*ss.pearsonr(df.preference, df.predicted)))

    print('\tSemantic change correlation:')
    print('\t\tSpearman r = {:.4f}, P = {:.4g}'
          .format(*ss.spearmanr(df.preference, df.sem_change)))
    print('\t\tPearson rho = {:.4f}, P = {:.4g}'
          .format(*ss.pearsonr(df.preference, df.sem_change)))

    plt.figure()
    plt.scatter(df.preference, df.predicted, alpha=0.3)
    plt.title(strain)
    plt.xlabel('Preference')
    plt.ylabel('Grammaticality')
    plt.savefig('figures/combinatorial_fitness_grammar_{}_{}.png'
                .format(args.namespace, strain), dpi=300)
    plt.close()

    plt.figure()
    plt.scatter(df.preference, df.sem_change, alpha=0.3)
    plt.title(strain)
    plt.xlabel('Preference')
    plt.ylabel('Semantic change')
    plt.savefig('figures/combinatorial_fitness_semantics_{}_{}.png'
                .format(args.namespace, strain), dpi=300)
    plt.close()

def analyze_semantics(args, model, vocabulary, seq_to_mutate, escape_seqs,
                      min_pos=None, max_pos=None, prob_cutoff=0., beta=1.,
                      comb_batch=None, plot_acquisition=True,
                      plot_namespace=None, verbose=True):
    """
    Computes semantic change and grammatical fitness, and generates mutation maps.

    Args:
        args (argparse.Namespace): command line arguments
        model: neural network model to be used
        vocabulary (dict[string, int]): dictionary of the available amino acids
        seq_to_mutate (str): sequence to mutate
        escape_seqs (dict[str, list[dict]]): dictionary of known escape sequences to their metadata
        min_pos (int): minimum mutation position
        max_pos (int): maximum mutation position
        prob_cutoff (float): probability cutoff for the grammatical fitness
        beta (float): beta parameter
        comb_batch (int): batch size for combinatorial mutation analysis, defaults to the number of sequences
        plot_acquisition (bool): whether to plot acquisition maps
        plot_namespace (string): namespace prefix for output files
        verbose (bool): whether to use verbose setting
    Returns:
        tuple:
            seqs: Array of mutated sequences
            probs: Array of predicted mutation probabilities
            changes: Array of semantic changes relative to base
    """

    if plot_acquisition:
        dirname = ('target/{}/semantics/cache'.format(args.namespace))
        mkdir_p(dirname)
        if plot_namespace is None:
            plot_namespace = args.namespace

    y_pred = predict_sequence_prob(
        args, seq_to_mutate, vocabulary, model, verbose=verbose
    )

    if min_pos is None:
        min_pos = 0
    if max_pos is None:
        max_pos = len(seq_to_mutate) - 1

    # construct word position map
    word_pos_prob = {}
    for i in range(min_pos, max_pos + 1):
        for word in vocabulary:
            if seq_to_mutate[i] == word:
                continue
            word_idx = vocabulary[word]
            prob = y_pred[i + 1, word_idx]
            word_pos_prob[(word, i)] = prob

    prob_seqs = { seq_to_mutate: [ { 'word': None, 'pos': None } ] }
    seq_prob = {}
    for (word, pos), prob in word_pos_prob.items():
        mutable = seq_to_mutate[:pos] + word + seq_to_mutate[pos + 1:]
        seq_prob[mutable] = prob
        if prob >= prob_cutoff:
            prob_seqs[mutable] = [ { 'word': word, 'pos': pos } ] # store data about sequences with sufficient probability

    seqs = np.array([ str(seq) for seq in sorted(seq_prob.keys()) ])

    # write mutations text file
    if plot_acquisition:
        ofname = dirname + '/{}_mutations.txt'.format(args.namespace)
        with open(ofname, 'w') as of:
            of.write('orig\tmutant\n')
            for seq in seqs:
                try:
                    didx = [
                        c1 != c2 for c1, c2 in zip(seq_to_mutate, seq)
                    ].index(True)
                    of.write('{}\t{}\t{}\n'
                             .format(didx, seq_to_mutate[didx], seq[didx]))
                except ValueError:
                    of.write('NA\n')

    base_embedding = embed_seqs(
        args, model, { seq_to_mutate: [ {} ] }, vocabulary,
        use_cache=False, verbose=False
    )[seq_to_mutate][0]['embedding']

    if comb_batch is None:
        comb_batch = len(seqs)
    n_batches = math.ceil(float(len(seqs)) / comb_batch)

    seq_change = {}
    for batchi in range(n_batches):
        # split to batches
        start = batchi * comb_batch
        end = (batchi + 1) * comb_batch
        prob_seqs_batch = {
            seq: prob_seqs[seq] for seq in seqs[start:end]
            if seq != seq_to_mutate
        }
        prob_seqs_batch = embed_seqs(
            args, model, prob_seqs_batch, vocabulary,
            use_cache=False, verbose=False
        )
        # calculate semantic changes
        for mut_seq in prob_seqs_batch:
            meta = prob_seqs_batch[mut_seq][0]
            sem_change = abs(base_embedding - meta['embedding']).sum()
            seq_change[mut_seq] = sem_change

    # cache semantic changes
    cache_fname = dirname + (
        '/analyze_semantics_{}_{}_{}.txt'
        .format(plot_namespace, args.model_name, args.dim)
    )
    probs, changes = [], []
    with open(cache_fname, 'w') as of:
        fields = [ 'pos', 'wt', 'mut', 'prob', 'change',
                   'is_viable', 'is_escape' ]
        of.write('\t'.join(fields) + '\n')
        for seq in seqs:
            prob = seq_prob[seq]
            change = seq_change[seq]
            mut = prob_seqs[seq][0]['word']
            pos = prob_seqs[seq][0]['pos']
            orig = seq_to_mutate[pos]
            is_viable = seq in escape_seqs
            is_escape = ((seq in escape_seqs) and
                         (sum([ m['significant']
                                for m in escape_seqs[seq] ]) > 0))
            fields = [ pos, orig, mut, prob, change, is_viable, is_escape ]
            of.write('\t'.join([ str(field) for field in fields ]) + '\n')
            probs.append(prob)
            changes.append(change)

    if plot_acquisition:
        from cached_semantics import cached_escape
        cached_escape(cache_fname, beta,
                      plot=plot_acquisition,
                      namespace=plot_namespace)

    return seqs, np.array(probs), np.array(changes)

def analyze_reinfection(
        args, model, seqs, vocabulary, wt_seq, mutants,
        namespace='reinfection',
):
    """
    Computes and writes semantic scores for reinfection scenarios.

    Args:
        args (argparse.Namespace): command line arguments
        model (LanguageModel): the model instance
        seqs (dict): dictionary of sequences
        vocabulary (dict[str, int]): dictionary of residues to a unique index
        wt_seq(str): the wild-type sequence
        mutants (dict[int, list[str]]): maps number of mutations to list of mutant sequences.
        namespace (str): the namespace to use to save files

    Returns:
        None
    """
    assert(len(mutants) == 1)
    n_mutations = list(mutants.keys())[0]

    # Compute mutational probabilities of base sequence.
    y_pred = predict_sequence_prob(
        args, wt_seq, vocabulary, model, verbose=False
    )
    word_pos_prob = {}
    for i in range(len(wt_seq)):
        for word in vocabulary:
            if wt_seq[i] == word:
                continue
            word_idx = vocabulary[word]
            prob = y_pred[i + 1, word_idx]
            word_pos_prob[(word, i)] = prob

    # Embed base sequence.
    base_embedding = embed_seqs(
        args, model, { wt_seq: [ {} ] }, vocabulary,
        use_cache=False, verbose=False
    )[wt_seq][0]['embedding']

    dirname = ('target/{}/reinfection/cache'.format(args.namespace))
    mkdir_p(dirname)
    fname = dirname + '/{}_mut_{}.txt'.format(args.namespace,
                                              namespace)

    # Compute mutant statistics.

    with open(fname, 'w') as of:
        for mutant in mutants[n_mutations]:
            positions = [ pos for pos in range(len(wt_seq))
                          if wt_seq[pos] != mutant[pos] ]
            assert(len(positions) == n_mutations)
            mut_str, raw_probs = '', []
            for pos in positions:
                word = mutant[pos]
                mut_str += wt_seq[pos] + str(pos + 1) + word + ','
                raw_probs.append(word_pos_prob[(word, pos)])
            if len(raw_probs) == 0:
                seq_prob = 0.
            else:
                seq_prob = np.mean(np.log10(raw_probs))

            embedding = embed_seqs(
                args, model, { mutant: [ {} ] }, vocabulary,
                use_cache=False, verbose=False
            )[mutant][0]['embedding']
            sem_change = abs(base_embedding - embedding).sum()

            fields = [ mut_str.rstrip(','), n_mutations,
                       seq_prob, sem_change ]
            of.write('\t'.join([ str(field) for field in fields ]) + '\n')

    # Compare to surveilled sequences.

    seqs = embed_seqs(args, model, seqs, vocabulary, use_cache=True)

    with open(fname, 'a') as of:
        for seq in seqs:
            meta = seqs[seq][0]
            if meta['strain'] != 'SARS-CoV-2' and \
               'hCoV-19' not in meta['strain']:
                continue
            if seq == wt_seq:
                continue
            if len(seq) != len(wt_seq):
                continue

            mut_str, raw_probs = '', []
            for pos in range(len(wt_seq)):
                if seq[pos] == wt_seq[pos]:
                    continue
                mut_str += wt_seq[pos] + str(pos + 1) + seq[pos] + ','
                raw_probs.append(word_pos_prob[(seq[pos], pos)])
            seq_prob = np.mean(np.log10(raw_probs))
            if not np.isfinite(seq_prob):
                continue

            sem_change = abs(base_embedding - meta['embedding']).sum()
            if not np.isfinite(sem_change):
                continue

            fields = [ mut_str.rstrip(','), len(raw_probs),
                       seq_prob, sem_change ]
            of.write('\t'.join([ str(field) for field in fields ]) + '\n')

def null_combinatorial_fitness(
        args, model, seqs, vocabulary, wt_seq, mutants,
        n_permutations=1000000, comb_batch=100, namespace=None,
):
    """
    Generates a null distribution, and print P-Values.

    Args:
        args (argparse.Namespace): command-line arguments
        model (LanguageModel): the model object
        seqs (dict): dictionary of sequences
        vocabulary (dict[str, int]): dictionary of residues to a unique index
        wt_seq (str): wild-type sequence
        mutants (dict[int, list[str]]): maps number of mutations to list of mutant sequences.
        n_permutations (int): number of null-draw permutations (defaults to 1000000)
        comb_batch (int): number of null-draw permutations per batch (defaults to 100)
        namespace (str): namespace to use, defaults to args.namespace

    Returns:
        None
    """
    if namespace is None:
        namespace = args.namespace

    assert(len(mutants) == 1)
    n_mutations = list(mutants.keys())[0]

    dirname = ('target/{}/combinatorial/cache'.format(args.namespace))
    mkdir_p(dirname)
    fname = dirname + '/{}_mut_{}.txt'.format(args.namespace,
                                              n_mutations)

    y_pred = predict_sequence_prob(
        args, wt_seq, vocabulary, model, verbose=False
    )

    word_pos_prob = {}
    for i in range(len(wt_seq)):
        for word in vocabulary:
            if wt_seq[i] == word:
                continue
            word_idx = vocabulary[word]
            prob = y_pred[i + 1, word_idx]
            word_pos_prob[(word, i)] = prob

    seq_prob = {}
    for mutant in mutants[n_mutations]:
        positions = [ pos for pos in range(len(wt_seq))
                      if wt_seq[pos] != mutant[pos] ]
        assert(len(positions) == n_mutations) # number of positions should match number of mutations
        mut_str, raw_probs = '', []
        for pos in positions:
            word = mutant[pos]
            mut_str += wt_seq[pos] + str(pos + 1) + word + ','
            raw_probs.append(word_pos_prob[(word, pos)])
        seq_prob[mut_str.rstrip(',')] = sum(np.log10(raw_probs))

    # Construct null.
    null, mut_strs = [], []
    for _ in range(n_permutations):
        positions = np.random.choice(len(wt_seq), n_mutations,
                                     replace=False)
        mut_str, raw_probs = '', []
        for pos in positions:
            choices = [ w for w in vocabulary if w != wt_seq[pos] ]
            word = np.random.choice(choices)
            mut_str += wt_seq[pos] + str(pos + 1) + word + ','
            raw_probs.append(word_pos_prob[(word, pos)])
        null.append(sum(np.log10(raw_probs)))
        mut_strs.append(mut_str.rstrip(','))
    null = np.array(null)

    for mut_str in seq_prob:
        p = sum(null >= seq_prob[mut_str]) / len(null)
        if p == 0:
            tprint('Mutant {}, P < {}'.format(mut_str, 1 / len(null)))
        else:
            tprint('Mutant {}, P = {}'.format(mut_str, p))
        for idx in np.where(null >= seq_prob[mut_str])[0]:
            tprint('\tMutant {} is fitter'.format(mut_strs[idx]))
