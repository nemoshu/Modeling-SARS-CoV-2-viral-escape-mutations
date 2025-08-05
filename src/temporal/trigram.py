class Trigram:
    """
    A trigram is a sequence of three consecutive amino acids in a strain.
    strain_pos is defined as the position of the first amino acid in the strain.
    """
    def __init__(self, amino_acids, strain_pos):
        """
        Initializes a Trigram object.

        Args:
            amino_acids (string): A list of amino acids.
            strain_pos (int): The position of the first amino acid.
        """
        self.amino_acids = amino_acids
        self.strain_pos = strain_pos

    def contains_position(self, pos):
        """
        Returns True if one of the amino acids in this trigram is from the
        given pos in the strain.

        A trigram covers positions [strain_pos, strain_pos + 2]

        Args:
            pos (int): The position of the first amino acid.

        Returns:
            contains_position (bool): whether one of the amino acids is from the given pos in the strain
        """
        return self.strain_pos <= pos and pos < self.strain_pos + len(self.amino_acids)