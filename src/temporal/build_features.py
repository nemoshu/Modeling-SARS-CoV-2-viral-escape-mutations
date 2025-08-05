import sys, os
import pandas as pd
import random
import numpy as np

from trigram import Trigram
import validation


def sample_strains(strains_by_year, num_of_samples):
    """
    Randomly picks num_of_samples strains from each year, sampling is done with replacement.

    Args:
        strains_by_year (list[list[string]]): a 2D list of strings where each sublist contains strain strings from a specific year
        num_of_samples (int): number of samples to pick from each year

    Returns:
        sampled_strains_by_year(list[list[string]]): a 2D list of strings containing the selected strains
    """
    sampled_strains_by_year = []

    for year_strains in strains_by_year:
        sampled_strains_by_year.append(random.choices(year_strains, k=num_of_samples))

    return sampled_strains_by_year


def sample_strains_cluster(strains_by_year, num_of_samples):
    """
    Picks the first num_of_samples strains from each year, sampling is done with replacement.

    Args:
        strains_by_year (list[list[string]]): a 2D list of strings containing the strains in each year
        num_of_samples (int): number of samples to pick from each year

    Returns:
        sampled_strains_by_year(list[list[string]]): a 2D list of strings containing the selected strains
    """

    sampled_strains_by_year = []
    # i = random.randint(0, 100)
    for year_strains in strains_by_year:
        sampled_strains_by_year.append(year_strains[:num_of_samples])

    return sampled_strains_by_year


def split_to_trigrams(strains_by_year, overlapping=True):
    """
    Splits the strains into trigrams, by default overlapping.
    If non-overlapping approach is used, the last amino acids are padded to make
    an extra trigram if the strain length is not evenly divisible by three.

    Args:
        strains_by_year (list): a 2D [year] [strain] list of strings,
        overlapping (bool): whether to use overlapping approach
    Returns:
        trigrams_by_year (list): a 3d [year] [strain] [trigram] list of Trigram objects.
    """

    if overlapping:
        step_size = 1
        num_of_trigrams = len(strains_by_year[0][0]) - 2
    else:
        step_size = 3
        num_of_trigrams = len(strains_by_year[0][0]) // step_size

    trigrams_by_year = []
    for year_strains in strains_by_year:
        year_trigrams = []

        for strain in year_strains:
            strain_trigrams = []

            for i in range(num_of_trigrams):
                pos = i * step_size # start position of trigram
                trigram = Trigram(strain[pos:pos + 3], pos)
                strain_trigrams.append(trigram)

            # for non-overlapping approach with remainders, perform padding when required
            remainder = len(strain) % step_size
            if remainder > 0:
                padding = '-' * (3 - remainder)
                amino_acids = strain[-remainder:] + padding
                trigram = Trigram(amino_acids, len(strain) - remainder)
                strain_trigrams.append(trigram)

            year_trigrams.append(strain_trigrams)

        trigrams_by_year.append(year_trigrams)

    return trigrams_by_year


def make_triplet_strains(strains_by_year, positions):
    """
    Splits each strain into substrings of 'triplets' referring to 3 overlapping
    trigrams (5 amino acids), centered at the given positions.

    Args:
        strains_by_year (list): a 2D [year, strain] list of strings,
        positions (list): list of centers
    Returns:
        triplet_strains_by_year (list): a 2d [year, strain] list of strings.
    """
    triplet_strains_by_year = []
    triplet_strain_margin = 2

    for strains_in_year in strains_by_year:
        triplet_strains_in_year = []
        for strain in strains_in_year:
            for p in positions:
                # pad to the front if smaller than the margin
                if p < triplet_strain_margin:
                    padding_size = triplet_strain_margin - p
                    triplet_strain = '-' * padding_size + strain[:p + triplet_strain_margin + 1]
                # otherwise pad to the back
                elif p > len(strain) - 1 - triplet_strain_margin:
                    padding_size = p - (len(strain) - 1 - triplet_strain_margin)
                    triplet_strain = strain[p - triplet_strain_margin:] + '-' * padding_size
                # no padding required
                else:
                    triplet_strain = strain[p - triplet_strain_margin:p + triplet_strain_margin + 1]
                triplet_strains_in_year.append(triplet_strain)
        triplet_strains_by_year.append(triplet_strains_in_year)

    return triplet_strains_by_year


def make_triplet_labels(triplet_strains_by_year):
    """
    Creates labels indicating whether the center amino acid in each triplet.
    Labels 1 if the center amino acid (position 2 in the 5-amino-acid triplet)
    differs between the last and penultimate year; 0 otherwise.

    Args:
        triplet_strains_by_year (list): a 2d [year, strain] list of strings.
    Returns:
        labels (list[int]): list of labels
    """
    num_of_triplets = len(triplet_strains_by_year[0])
    epitope_position = 2

    labels = []
    for i in range(num_of_triplets):
        if triplet_strains_by_year[-1][i][epitope_position] == triplet_strains_by_year[-2][i][epitope_position]:
            labels.append(0) # no mutation
        else:
            labels.append(1) # mutation

    return labels


def get_majority_baselines(triplet_strains_by_year, labels):
    """
    Returns accuracy, precision, recall, f1-score and mcc for the baseline
    approach of simply predicting mutation epitope in the last year differs
    from the majority one.

    Majority is determined from all years except the last.

    Args:
        triplet_strains_by_year (list): a 2d [year, strain] list of triplet strings.
        labels (list): list of labels
    Returns:
        acc (float): accuracy
        precision (float): precision
        recall (float): recall
        f1score (float): f1-score
        mcc (float): mcc
    """
    epitope_position = 2

    predictions = [] # predicted labels - 0 if matches majority epitope; 1 otherwise
    for i in range(len(labels)):
        # count and find majority epitope
        epitopes = []
        for year in range(len(triplet_strains_by_year) - 1):
            epitopes.append(triplet_strains_by_year[year][i][epitope_position])
        majority_epitope = max(set(epitopes), key=epitopes.count)

        if triplet_strains_by_year[-2][i][epitope_position] == majority_epitope:
            predictions.append(0)
        else:
            predictions.append(1)

    # obtain confusion matrix and results
    conf_matrix = validation.get_confusion_matrix(np.array(labels), np.array(predictions))
    acc = validation.get_accuracy(conf_matrix)
    precision = validation.get_precision(conf_matrix)
    recall = validation.get_recall(conf_matrix)
    f1score = validation.get_f1score(conf_matrix)
    mcc = validation.get_mcc(conf_matrix)

    return acc, precision, recall, f1score, mcc


def extract_positions_by_year(positions, trigrams_by_year):
    """
    Extracts trigrams that contain an amino acid which overlaps one of the given positions.
    Expects and returns a 3d [year] [strain] [trigram] list of Trigram objects.

    Args:
        positions (list): List of positions
        trigrams_by_year (list): A 3d [year] [strain] [trigram] list of Trigram objects.

    Returns:
        extracted_by_year (list): Extracted 3d [year] [strain] [trigram] list of Trigram objects.
    """
    strain = trigrams_by_year[0][0]
    strain_idxs_to_extract = []
    idx = 0

    for pos in positions:

        # loop until a trigram contains pos
        pos_found = False
        while not pos_found:
            trigram = strain[idx]
            if trigram.contains_position(pos):
                pos_found = True
            else:
                idx += 1

        # extract remaining indexes in trigram
        pos_extracted = False
        while not pos_extracted:
            trigram = strain[idx]
            if trigram.contains_position(pos):
                strain_idxs_to_extract.append(idx)
                idx += 1
            else:
                pos_extracted = True

    def extract_idxs(strain_trigrams):
        """
        Helper function to extract trigrams with the specified indexes.

        Args:
            strain_trigrams (list): List of Trigram objects

        Returns:
            strain_trigrams_extracted (list): List of Trigram objects extracted
        """
        return [strain_trigrams[i] for i in strain_idxs_to_extract]

    # extract for each year
    extracted_by_year = []
    for year_trigrams in trigrams_by_year:
        extracted_by_year.append(list(map(extract_idxs, year_trigrams)))

    return extracted_by_year


def squeeze_trigrams(trigrams_by_year):
    """
    Takes a 3d [year, strain, trigram] list and squeezes the 2nd dimension to return a 2d list [year, trigram].

    Args:
        trigrams_by_year (list): 3d [year, strain, trigram] list of Trigram objects.

    Returns:
        squeezed_trigrams_by_year (list): Squeezed 2d [year, trigram] list of Trigram objects.
    """
    squeezed_trigrams_by_year = []

    for year_trigrams in trigrams_by_year:
        squeezed_trigrams = []

        for trigrams in year_trigrams:
            squeezed_trigrams += trigrams # squeeze by simple concatenation

        squeezed_trigrams_by_year.append(squeezed_trigrams)

    return squeezed_trigrams_by_year


def replace_uncertain_amino_acids(amino_acids):
    """
    Randomly selects replacements for all uncertain amino acids.
    Expects and returns a string.

    Args:
        amino_acids (string): uncertain amino acid string

    Returns:
        amino_acids (string): replaced amino acid string
    """
    replacements = {'B': 'DN',
                    'J': 'IL',
                    'Z': 'EQ',
                    'X': 'ACDEFGHIKLMNPQRSTVWY'} # valid random replacements

    for uncertain in replacements.keys():
        amino_acids = amino_acids.replace(uncertain, random.choice(replacements[uncertain]))

    return amino_acids


def map_trigrams_to_idxs(nested_trigram_list, trigram_to_idx):
    """
    Takes a nested list containing Trigram objects and maps them to their index.

    Args:
        nested_trigram_list (list): nested list of Trigram objects
        trigram_to_idx (dict[string, int]): dictionary of Trigram Amino Acids mapped to integer indexes

    Returns:
        mapped (list): trigrams indexes
    """
    dummy_idx = len(trigram_to_idx)

    def mapping(trigram):
        """
        Helper function as a parameter to the map function.
        Maps a Trigram object to its corresponding index.

        Args:
            trigram (string): string of trigram amino acids

        Returns:
            index (int): Trigram index, or dummy index of len(trigram_to_idx) if unavailable
        """
        if isinstance(trigram, Trigram):
            trigram.amino_acids = replace_uncertain_amino_acids(trigram.amino_acids)

            if '-' not in trigram.amino_acids:
                return trigram_to_idx[trigram.amino_acids]
            else:
                return dummy_idx

        # if not reached individual trigram level, recurse
        elif isinstance(trigram, list):
            return list(map(mapping, trigram))

        else:
            raise TypeError('Expected nested list of Trigrams, but encountered {} in recursion.'.format(type(trigram)))

    return list(map(mapping, nested_trigram_list))


def map_idxs_to_vecs(nested_idx_list, idx_to_vec):
    """
    Takes a nested list of indexes and maps them to their trigram vec (np array).

    Args:
        nested_idx_list (list): nested list of indexes
        idx_to_vec (nparray): index to vector mapping

    Returns:
        mapped (list): vector mapping of indexes
    """
    # represent the 3-grams containing '-' by zero vector in ProVect
    # dummy_vec = np.array([0] * idx_to_vec.shape[1])

    # represent the 3-grams containing '-' by 'unknown' vector in ProVect
    dummy_vec = idx_to_vec[idx_to_vec.shape[0] - 1]

    def mapping(idx):
        """
        Helper function as a parameter to the map function.
        Maps an index to a vector.

        Args:
            idx (int): index
        Returns:
            vec (nparray): vector mapping corresponding to the index, or dummy vector of ``idx_to_vec[idx_to_vec.shape[0]-1]`` if unavailable.
        """
        if isinstance(idx, int):
            if idx < idx_to_vec.shape[0]:
                return idx_to_vec[idx]
            else:
                return dummy_vec

        elif isinstance(idx, list):
            return list(map(mapping, idx)) # if not yet individual index level, recurse

        else:
            raise TypeError('Expected nested list of ints, but encountered {} in recursion.'.format(type(idx)))

    return list(map(mapping, nested_idx_list))


def get_diff_vecs(trigram_vecs_by_year):
    """
    Calculates the elementwise difference between each consecutive trigram vec.

    Args:
        trigram_vecs_by_year (nparray): list of Trigram objects in each year

    Returns:
        diff_vecs_by_year (nparray): elementwise differences in each year
    """
    diff_vecs_by_year = np.zeros(
        (trigram_vecs_by_year.shape[0] - 1, trigram_vecs_by_year.shape[1], trigram_vecs_by_year.shape[2]))
    for i in range(diff_vecs_by_year.shape[0]):
        diff_vecs_by_year[i] = trigram_vecs_by_year[i + 1] - trigram_vecs_by_year[i]

    return diff_vecs_by_year


def indexes_to_mutations(trigram_indexes_x, trigram_indexes_y):
    """
    Creates a numpy array containing 1's in positions where trigram_indexes_x and
    trigram_indexes_y differ, corresponding to mutated sites and zeros elsewhere.

    Args:
        trigram_indexes_x (list): first list of trigram indexes
        trigram_indexes_y (list): second list of trigram indexes
    Returns:
        mutated_indexes_x (nparray): list denoting whether there is mutation at that index
    """
    assert (len(trigram_indexes_x) == len(trigram_indexes_y)) # list lengths must match

    mutations = np.zeros(len(trigram_indexes_x)) # all zeros
    for i in range(len(trigram_indexes_x)):
        if trigram_indexes_x[i] != trigram_indexes_y[i]: # set any non-matching instances to 1
            mutations[i] = 1

    return mutations


def reshape_to_linear(vecs_by_year, window_size=3):
    """
    Reshapes vectors to linear by concatenating vectors from the last <window_size> years.

    Args:
        vecs_by_year (list): list of vectors in each year
        window_size (int): window size, i.e., number of years to look back

    Returns:
        reshaped (list): reshaped linear list
    """
    reshaped = [[]] * len(vecs_by_year[0])

    for year_vecs in vecs_by_year[-window_size:]:
        for i, vec in enumerate(year_vecs):
            reshaped[i] = reshaped[i] + vec.tolist()

    return reshaped
