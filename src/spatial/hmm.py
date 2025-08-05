from hmmlearn.hmm import MultinomialHMM

def fit(seqs, n_components=1):
    """
    Wraps a MultinomialHMM.

    Args:
        seqs (list): (unused) list of sequences
        n_components (int): number of states
    """
    MultinomialHMM(
        n_components=n_components,
        startprob_prior=1.0,
        transmat_prior=1.0,
        algorithm='viterbi',
        random_state=1,
        n_iter=100,
        tol=0.01,
        verbose=True,
        params='ste',
        init_params='ste'
    )
