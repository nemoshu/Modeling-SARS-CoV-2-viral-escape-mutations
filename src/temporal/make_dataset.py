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