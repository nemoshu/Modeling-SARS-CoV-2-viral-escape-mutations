from anndata import AnnData
from collections import Counter
import datetime
from dateutil.parser import parse as dparse
import errno
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import scanpy as sc
from scipy.sparse import csr_matrix, dok_matrix
import scipy.stats as ss
import seaborn as sns
import sys
import time
import warnings

from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
from Bio import Seq, SeqIO

np.random.seed(1)
random.seed(1)

def tprint(string):
    """
    Prints content to the standard output with timestamp.

    Args:
        string (str): content to print
    """
    string = str(string)
    sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    sys.stdout.write(string + '\n')
    sys.stdout.flush()

def mkdir_p(path):
    """
    Makes a directory if it doesn't exist.

    Args:
        path (str): directory to create
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def iterate_lengths(lengths, seq_len):
    """
    Generates start and end indices for each segment.

    Args:
        lengths (list[int]): list of segment lengths
        seq_len (int): total sequence length, used for error checking

    Yields:
        Tuple[int, int]: (start index, end index) for each segment

    Warns:
        Generates warning if the length of one segment is longer than the expected sequence length
    """
    curr_idx = 0
    for length in lengths:
        if length > seq_len:
            sys.stderr.write(
                'Warning: length {} greater than expected '
                'max length {}\n'.format(length, seq_len)
            )
        yield (curr_idx, curr_idx + length)
        curr_idx += length
