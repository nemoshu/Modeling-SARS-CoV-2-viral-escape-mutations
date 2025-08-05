from utils import *

from sklearn.metrics import auc

def compute_p(true_val, n_interest, n_total, n_permutations=10000):
    """
    Helper function. Computes the corresponding p-value for an observed AUC value by permutation testing.

    Args:
        true_val (float): Observed normalised AUC value to test for significance
        n_interest (int): Number of items of interest
        n_total (int): Total number of items in dataset
        n_permutations (int, optional): Number of permutations for the null distribution. Defaults to 10000.

    Returns:
        p (float): P-value estimating the probability that the observed AUC (true-val) would occur.
    """

    # Stimulates a null distribution
    null_distribution = []
    norm = n_interest * n_total
    for _ in range(n_permutations):
        interest = set(np.random.choice(n_total, size=n_interest,
                                        replace=False))
        n_acquired = 0
        acquired, total = [], []
        for i in range(n_total):
            if i in interest:
                n_acquired += 1
            acquired.append(n_acquired) # counts acquired items
            total.append(i + 1)
        null_distribution.append(auc(total, acquired) / norm)
    null_distribution = np.array(null_distribution) # converts to np.array format for sum
    return sum(null_distribution >= true_val) / n_permutations

def cached_escape(cache_fname, beta, plot=True, namespace='semantics'):
    """
    Generates escape prediction maps and discovery curves, and evaluates escape mutation detection performance.

    Args:
        cache_fname (str): Path to a cached file containing model outputs.
        beta (float): Beta parameter - weighting factor in the acquisition score.
        plot (bool, optional): Whether to generate plots. Defaults to True.
        namespace (str, optional): Prefix for output filenames. Defaults to 'semantics'.
    """

    # reading and parsing the cached file
    prob, change, escape_idx, viable_idx = [], [], [], []
    with open(cache_fname) as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split('\t')
            pos = int(fields[0]) # mutation position
            prob.append(float(fields[3])) # grammaticality
            change.append(float(fields[4])) # semantic change
            viable_idx.append(fields[5] == 'True') # mutation is visible
            escape_idx.append(fields[6] == 'True') # mutation is anti-body escape

    # converting to numpy formats for the scipy module
    prob, orig_prob = np.array(prob), np.array(prob)
    change, orig_change  = np.array(change), np.array(change)
    escape_idx = np.array(escape_idx)
    viable_idx = np.array(viable_idx)

    # calculating acquisition scores
    acquisition = ss.rankdata(change) + (beta * ss.rankdata(prob))

    # boolean array for whether there is non-zero semantic shift
    pos_change_idx = change > 0

    # boolean array for escape mutations that also had semantic change
    pos_change_escape_idx = np.logical_and(pos_change_idx, escape_idx)

    escape_prob = prob[pos_change_escape_idx] # escape probability list
    escape_change = change[pos_change_escape_idx] # semantic change list for escaping mutations
    prob = prob[pos_change_idx] # probability of each change happening
    change = change[pos_change_idx] # all non-zero semantic changes

    # applying log10 for plots
    log_prob, log_change = np.log10(prob), np.log10(change)
    log_escape_prob, log_escape_change = (np.log10(escape_prob),
                                          np.log10(escape_change))

    if plot:
        # plotting acquisition graph
        plt.figure()
        plt.scatter(log_prob, log_change, c=acquisition[pos_change_idx],
                    cmap='viridis', alpha=0.3)
        plt.scatter(log_escape_prob, log_escape_change, c='red',
                    alpha=0.5, marker='x')
        plt.xlabel(r'$ \log_{10}(\hat{p}(x_i | \mathbf{x}_{[N] ∖ \{i\} })) $')
        plt.ylabel(r'$ \log_{10}(\Delta \mathbf{\hat{z}}) $')
        plt.savefig('figures/{}_acquisition.png'
                    .format(namespace), dpi=300)
        plt.close()

        # plotting a random acquisition graph
        rand_idx = np.random.choice(len(prob), len(escape_prob))
        plt.figure()
        plt.scatter(log_prob, log_change, c=acquisition[pos_change_idx],
                    cmap='viridis', alpha=0.3)
        plt.scatter(log_prob[rand_idx], log_change[rand_idx], c='red',
                    alpha=0.5, marker='x')
        plt.xlabel(r'$ \log_{10}(\hat{p}(x_i | \mathbf{x}_{[N] ∖ \{i\} })) $')
        plt.ylabel(r'$ \log_{10}(\Delta \mathbf{\hat{z}}) $')
        plt.savefig('figures/{}_acquisition_rand.png'
                    .format(namespace), dpi=300)
        plt.close()

    if len(escape_prob) == 0:
        print('No escape mutations found.')
        return

    acq_argsort = ss.rankdata(-acquisition) # sorts acquisition in descending sequence (most acquisition first)
    escape_rank_dist = acq_argsort[escape_idx] # escape distances ranked of actual escape mutations

    # printing statistics from escape sequences
    size = len(prob)
    print('Number of escape seqs: {} / {}'
          .format(len(escape_rank_dist), sum(escape_idx)))
    print('Mean rank: {} / {}'.format(np.mean(escape_rank_dist), size))
    print('Median rank: {} / {}'.format(np.median(escape_rank_dist), size))
    print('Min rank: {} / {}'.format(np.min(escape_rank_dist), size))
    print('Max rank: {} / {}'.format(np.max(escape_rank_dist), size))
    print('Rank stdev: {} / {}'.format(np.std(escape_rank_dist), size))

    max_consider = len(prob)
    n_consider = np.array([ i + 1 for i in range(max_consider) ]) # counter array (X-axis) for max_consider ([1, 2, ..., max_consider])

    n_escape = np.array([ sum(escape_rank_dist <= i + 1)
                          for i in range(max_consider) ]) # number of escape mutations in the top i + 1 acquisitions
    norm = max(n_consider) * max(n_escape)
    norm_auc = auc(n_consider, n_escape) / norm # AUC (CSCS)

    escape_rank_prob = ss.rankdata(-orig_prob)[escape_idx]
    n_escape_prob = np.array([ sum(escape_rank_prob <= i + 1)
                               for i in range(max_consider) ])
    norm_auc_prob = auc(n_consider, n_escape_prob) / norm # AUC if ranking was only done by grammaticality

    escape_rank_change = ss.rankdata(-orig_change)[escape_idx]
    n_escape_change = np.array([ sum(escape_rank_change <= i + 1)
                                 for i in range(max_consider) ])
    norm_auc_change = auc(n_consider, n_escape_change) / norm # AUC if ranking was only done by semantic change

    # plotting discovery curves
    if plot:
        plt.figure()
        plt.plot(n_consider, n_escape)
        plt.plot(n_consider, n_escape_change, c='C0', linestyle='-.')
        plt.plot(n_consider, n_escape_prob, c='C0', linestyle=':')
        plt.plot(n_consider, n_consider * (len(escape_prob) / len(prob)),
                 c='gray', linestyle='--')

        plt.xlabel(r'$ \log_{10}() $')
        plt.ylabel(r'$ \log_{10}(\Delta \mathbf{\hat{z}}) $')

        plt.legend([
            r'$ \Delta \mathbf{\hat{z}} + ' +
            r'\beta \hat{p}(x_i | \mathbf{x}_{[N] ∖ \{i\} }) $,' +
            (' AUC = {:.3f}'.format(norm_auc)),
            r'$  \Delta \mathbf{\hat{z}} $ only,' +
            (' AUC = {:.3f}'.format(norm_auc_change)),
            r'$ \hat{p}(x_i | \mathbf{x}_{[N] ∖ \{i\} }) $ only,' +
            (' AUC = {:.3f}'.format(norm_auc_prob)),
            'Random guessing, AUC = 0.500'
        ])
        plt.xlabel('Top N')
        plt.ylabel('Number of escape mutations in top N')
        plt.savefig('figures/{}_consider_escape.png'
                    .format(namespace), dpi=300)
        plt.close()

    # printing escape semantics data
    print('Escape semantics, beta = {} [{}]'
          .format(beta, namespace))

    # computing the P value
    norm_auc_p = compute_p(norm_auc, sum(escape_idx), len(escape_idx))

    print('AUC (CSCS): {}, P = {}'.format(norm_auc, norm_auc_p))
    print('AUC (semantic change only): {}'.format(norm_auc_change))
    print('AUC (grammaticality only): {}'.format(norm_auc_prob))

    # testing whether escape mutations are more/less grammatical than others
    print('{:.4g} (mean log prob), {:.4g} (mean log prob escape), '
          '{:.4g} (p-value)'
          .format(log_prob.mean(), log_escape_prob.mean(),
                  ss.mannwhitneyu(log_prob, log_escape_prob,
                                  alternative='two-sided')[1]))

    # testing whether escape mutations are more/less semantically distinct
    print('{:.4g} (mean log change), {:.4g} (mean log change escape), '
          '{:.4g} (p-value)'
          .format(change.mean(), escape_change.mean(),
                  ss.mannwhitneyu(change, escape_change,
                                  alternative='two-sided')[1]))

if __name__ == '__main__':
    """
    Generates escape prediction maps and discovery curves, 
    and evaluates escape mutation detection performance.
    
    Command line argument: Path of Input File
    
    Input File: as specified in the command line argument
    Output Files: 
        - figures/semantics_acquisition.png
        - figures/semantics_acquisition_rand.png
        - figures/semantics_consider_escape.png
    """
    cached_escape(sys.argv[1], 1.)
