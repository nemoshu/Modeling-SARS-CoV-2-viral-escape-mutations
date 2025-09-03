import math
import ast

import cluster
import pandas as pd
import numpy as np
import make_dataset
import build_features
import random



def read_dataset(path, data_path, limit=0, concat=False):
    """
    Reads the data set from given path, expecting it to contain a 'y' column with
    the label and each year in its own column containing a number of trigram indexes.
    Limit sets the maximum number of examples to read, zero meaning no limit.
    If concat is true each of the trigrams in a year is concatenated, if false
    they are instead summed elementwise.

    Args:
        path (str): path to CSV dataset
        data_path (str): path to trigram vector data
        limit (int): maximum number of examples to read (0 means no limit)
        concat (bool): whether to concatenate trigram vectors for each year. False means to sum them.

    Returns:
        trigram_vecs (np.array): trigram vectors, shape [num_time_steps, num_samples, feature_dim]
        labels (ndarray): target labels, 1 for escape mutation and 0 otherwise - from df['y']
    """
    # subtype_flag, data_path = make_dataset.subtype_selection(subtype)
    _, trigram_vecs_data = make_dataset.read_trigram_vecs(data_path)

    df = pd.read_csv(path)

    if limit != 0:
        df = df.head(limit)

    labels = df['y'].values
    trigram_idx_strings = df.loc[:, df.columns != 'y'].values
    parsed_trigram_idxs = [list(map(lambda x: ast.literal_eval(x), example)) for example in trigram_idx_strings]
    trigram_vecs = np.array(build_features.map_idxs_to_vecs(parsed_trigram_idxs, trigram_vecs_data))

    if concat:
        trigram_vecs = np.reshape(trigram_vecs, [len(df.columns) - 1, len(df.index), -1])
    else:
        # Sum trigram vecs instead of concatenating them
        trigram_vecs = np.sum(trigram_vecs, axis=2)
        trigram_vecs = np.moveaxis(trigram_vecs, 1, 0)

    return trigram_vecs, labels


def get_time_string(time):
    """
    Creates a string representation of minutes and seconds from the given time.

    Args:
        time (int): time in seconds

    Returns:
        time_string (str): time string representation, e.g. '  2m  5s'
    """
    mins = time // 60
    secs = time % 60
    time_string = ''

    if mins < 10:
        time_string += '  '
    elif mins < 100:
        time_string += ' '

    time_string += '%dm ' % mins

    if secs < 10:
        time_string += ' '

    time_string += '%ds' % secs

    return time_string
