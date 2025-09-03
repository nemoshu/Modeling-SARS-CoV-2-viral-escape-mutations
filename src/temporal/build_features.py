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
