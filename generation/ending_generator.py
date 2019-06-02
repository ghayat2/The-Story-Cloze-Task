class EndingGenerator:
    def generate_endings(self, correct_ending, nb_samples):
        """
        :param correct_ending: A correct story ending sentence.
        :param nb_samples: Number of fake endings to generate.
        :return: nb_samples generated fake endings.
        """
        raise NotImplementedError("Subclass did not implement 'generate_ending'")
