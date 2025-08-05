import pandas as pd
import random
import math


def subtype_selection(subtype):
    """
    Translate a subtype in the string format into the corresponding integer subtype flag.

    Args:
        subtype (str): subtype to translate, from 'H1N1', 'H3N2', 'H5N1' or 'COV19'

    Returns:
        subtype_flag (int): subtype flag, mapping 'H1N1', 'H3N2', 'H5N1', 'COV19' to 0, 1, 2, 3 respectively
    """
    global subtype_flag, data_path
    if subtype == 'H1N1':
        subtype_flag = 0
    elif subtype == 'H3N2':
        subtype_flag = 1
    elif subtype == 'H5N1':
        subtype_flag = 2
    elif subtype == 'COV19':
        subtype_flag = 3

    return subtype_flag


def read_trigram_vecs(subtype):
    """
    Reads the csv file containing 100 dimensional prot vecs, the
    data_path argument indicating where it is located.
    Returns a dictionary that maps a 3gram of amino acids to its
    index and a numpy array containing the trigram vecs.

    Input file: protVec_100d_3grams.csv

    Args:
        subtype (any): unused

    Returns:
        trigram_to_idx (dict[list, int]): trigram to index map
        trigram_vec (dataframe): trigram vectors for each trigram, shaped (n_trigrams, 100)
    """
    data_path = '/Users/nemoshu/Computer science experiments/UCL/BiologyNLP/output/'
    prot_vec_file = 'protVec_100d_3grams.csv'

    df = pd.read_csv(data_path + prot_vec_file, delimiter='\t')
    trigrams = list(df['words'])
    trigram_to_idx = {trigram: i for i, trigram in enumerate(trigrams)}
    trigram_vecs = df.loc[:, df.columns != 'words'].values

    return trigram_to_idx, trigram_vecs


def read_strains_from(data_files, data_path):
    """
    Reads the raw strains from the data_files located by the data_path.
    Returns a pandas series for each data file, contained in an ordered list.

    Args:
        data_files (list): list of data file names
        data_path (str): directory containing the data files

    Returns:
        raw_strains (list): list of raw strains, where each element is a pandas series
    """
    # _, data_path = subtype_selection(subtype)
    raw_strains = []
    for file_name in data_files:
        df = pd.read_csv(data_path + file_name)
        strains = df['seq']
        raw_strains.append(strains)

    return raw_strains


def train_test_split_strains(strains_by_year, test_split, cluster):
    """
    Shuffles the strains in each year and splits them into two disjoint sets,
    of size indicated by the test_split.
    Expects and returns pandas dataframe or series.

    Args:
        strains_by_year (pandas.DataFrame): dataframe of strains by year
        test_split (float): percentage of data to use for testing
        cluster (string): cluster type

    Returns:
        train_strains (pandas.DataFrame): dataframe of training strains
        test_strains (pandas.DataFrame): dataframe of test strains
    """
    train_strains, test_strains = [], []
    # random cluster
    if cluster == 'random':
        for strains in strains_by_year:
            num_of_training_examples = int(math.floor(strains.count() * (1 - test_split)))
            shuffled_strains = strains.sample(frac=1).reset_index(drop=True)
            train = shuffled_strains.iloc[:num_of_training_examples].reset_index(drop=True)
            test = shuffled_strains.iloc[num_of_training_examples:].reset_index(drop=True)
            train_strains.append(train)
            test_strains.append(test)
    else:
        # change the starting index for the time-series training samples for multiple experiments
        for strains in strains_by_year:
            num_of_training_examples = int(math.floor(strains.count() * (1 - test_split)))
            train = strains.iloc[:800].reset_index(drop=True)
            test = strains.iloc[800:1000].reset_index(drop=True)
            train_strains.append(train)
            test_strains.append(test)
    return train_strains, test_strains



