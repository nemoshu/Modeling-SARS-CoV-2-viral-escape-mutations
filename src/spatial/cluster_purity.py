from utils import *

def print_purity(metas, entries):
    """
    Calculates and prints out cluster purity for the specified metadata fields

    Args:
        metas (dict): Dictionary of metadata for sequences, mapping sequences to their metadata dictionaries
        entries (list): List of metadata fields to evaluate

    Returns:
        None
    """
    for entry in entries:
        tprint('Calculating purity of {}'.format(entry))
        # calculating purities for each entry
        cluster2entry = {}  # cluster to entry dictionary
        for accession in metas:
            meta = metas[accession]
            try:
                cluster = meta['cluster']
            except:
                continue  # if there is no cluster data, skip
            if cluster not in cluster2entry:
                cluster2entry[cluster] = []
            cluster2entry[cluster].append(meta[entry])  # add entry to the appropriate key
        largest_pct_entry = []
        for cluster in cluster2entry:
            count = Counter(cluster2entry[cluster]).most_common(1)[0][1] # count the number of the most common element in the cluster
            pct = float(count) / len(cluster2entry[cluster]) # calculate purity percentage
            largest_pct_entry.append(pct)
            tprint('Cluster {}, largest % = {}'.format(cluster, pct))
        tprint('Purity, phylo clustering and {}: {}'
               .format(entry, np.mean(largest_pct_entry)))


def flu_purity(phylo_method='mafft'):
    """
    Calculates and prints cluster purity for influenza sequences

    Args:
        phylo_method(str, optional): Phylogenetic clustering method (e.g., 'mafft', 'clustalomega'). Defaults to 'mafft'.

    Returns:
        None
    """
    from flu import load_meta
    meta_fnames = [ 'data/influenza/ird_influenzaA_HA_allspecies_meta.tsv' ]
    metas = load_meta(meta_fnames)

    # selects the appropriate cluster filename.
    if phylo_method == 'mafft':
        cluster_fname = 'target/influenza/clusters/all.clusters_0.117.txt'
    elif phylo_method == 'mafft_sl':
        cluster_fname = 'target/influenza/clusters/all_singlelink_0.119.txt'
    elif phylo_method == 'clustalomega':
        cluster_fname = 'target/influenza/clusters/clustal_omega_clusters_0.382.txt'
    elif phylo_method == 'clustalomega_sl':
        cluster_fname = 'target/influenza/clusters/clustal_omega_singlelink_0.3.txt'
    elif phylo_method == 'mrbayes':
        cluster_fname = 'target/influenza/clusters/mrbayes_clusters_125.txt'
    elif phylo_method == 'mrbayes_sl':
        cluster_fname = 'target/influenza/clusters/mrbayes_singlelink_0.25.txt'
    elif phylo_method == 'raxml':
        cluster_fname = 'target/influenza/clusters/raxml_clusters_0.1.txt'
    elif phylo_method == 'raxml_sl':
        cluster_fname = 'target/influenza/clusters/raxml_singlelink_0.56.txt'
    elif phylo_method == 'fasttree':
        cluster_fname = 'target/influenza/clusters/fasttree_clusters_5.001.txt'
    elif phylo_method == 'fasttree_sl':
        cluster_fname = 'target/influenza/clusters/fasttree_singlelink_0.08.txt'
    else:
        raise ValueError('Invalid phylo method {}'.format(phylo_method))

    # reading and parsing each line of cluster file
    with open(cluster_fname) as f:
        f.readline() # skip header
        for line in f:
            if 'Reference_Perth2009' in line:
                continue
            fields = line.rstrip().split()
            if 'mafft' in phylo_method:
                accession = fields[0].split('_')[2]
                metas[accession]['cluster'] = fields[1]
            else:
                accession = fields[0].split('_')[1]
                metas[accession]['cluster'] = fields[1]

    print_purity(metas, [ 'Subtype', 'Host Species' ])

def hiv_purity(phylo_method='mafft'):
    """
    Calculates and prints cluster purity for influenza sequences

     Args:
        phylo_method(str, optional): Phylogenetic clustering method. Defaults to 'mafft'.
    """
    from hiv import load_meta
    meta_fnames = [ 'data/hiv/HIV-1_env_samelen.fa' ]
    metas = load_meta(meta_fnames)
    metas = { accession.split('.')[-1]: metas[accession]
              for accession in metas }

    # selects the appropriate cluster filename.
    if phylo_method == 'mafft':
        cluster_fname = 'target/hiv/clusters/all.clusters_0.445.txt'
    elif phylo_method == 'mafft_sl':
        cluster_fname = 'target/hiv/clusters/all_singlelink_0.445.txt'
    elif phylo_method == 'clustalomega':
        cluster_fname = 'target/hiv/clusters/clustal_omega_clusters_0.552.txt'
    elif phylo_method == 'clustalomega_sl':
        cluster_fname = 'target/hiv/clusters/clustal_omega_singelink_0.49.txt'
    elif phylo_method == 'mrbayes':
        cluster_fname = 'target/hiv/clusters/mrbayes_clusters_4.1.txt'
    elif phylo_method == 'mrbayes_sl':
        cluster_fname = 'target/hiv/clusters/mrbayes_singlelink_0.125.txt'
    elif phylo_method == 'raxml':
        cluster_fname = 'target/hiv/clusters/raxml_clusters_0.77.txt'
    elif phylo_method == 'raxml_sl':
        cluster_fname = 'target/hiv/clusters/raxml_singlelink_12.txt'
    elif phylo_method == 'fasttree':
        cluster_fname = 'target/hiv/clusters/fasttree_clusters_1.71.txt'
    elif phylo_method == 'fasttree_sl':
        cluster_fname = 'target/hiv/clusters/fasttree_singlelink_0.64.txt'
    else:
        raise ValueError('Invalid phylo method {}'.format(phylo_method))

    # reading and parsing each line of cluster file
    with open(cluster_fname) as f:
        f.readline() # skip header
        for line in f:
            fields = line.rstrip().split()
            if fields[0].endswith('NC_001802'):
                accession = 'NC_001802' # special case to avoid illogical split
            else:
                accession = fields[0].split('_')[-1]
            cluster = fields[1]
            metas[accession]['cluster'] = fields[1]

    print_purity(metas, [ 'subtype' ])

if __name__ == '__main__':
    """
    Runs cluster purity analysis for Flu and HIV.
    
    Command line argument: Phylo Method
    
    Input files:
        as specified in flu_purity() and hiv_purity() functions
    """
    tprint('Flu HA...')
    flu_purity(sys.argv[1])

    tprint('HIV Env...')
    hiv_purity(sys.argv[1])
