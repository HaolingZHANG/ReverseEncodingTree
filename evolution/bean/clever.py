import numpy


class Learner(object):

    def __init__(self, pre_dataset):
        """
        Initialize the gradient collector.

        :param pre_dataset: pre-dataset for learning how to generate the attacker.
        """
        self.saved_individuals = None
        self.gradients_collector = []
        self.pre_dataset = pre_dataset

    def collect_gradients(self, current_individuals):
        """
        Calculate and collect the gradients from the previous and current NeuroEvolution.

        :param current_individuals: current individuals of NeuroEvolution.
        """
        current_gradients = []

        if self.saved_individuals is not None:
            for s_i, c_i in zip(self.saved_individuals, current_individuals):
                current_gradients.append(((c_i.fitness - s_i.fitness) / (numpy.array(c_i.feature_matrix) -
                                                                         numpy.array(s_i.feature_matrix)).tolist()))
        else:
            for c_i in current_individuals:
                current_gradients.append((c_i.fitness / numpy.array(c_i.feature_matrix)).tolist())

        norm_one = numpy.linalg.norm(numpy.array(current_gradients), ord=1, keepdims=True)
        norm_two = numpy.linalg.norm(numpy.array(current_gradients), ord=2, keepdims=True)
        norm_inf = numpy.linalg.norm(numpy.array(current_gradients), ord=numpy.inf, keepdims=True)

        self.gradients_collector.append(current_gradients)
        self.saved_individuals = []

    def insert_previous(self, saved_individual):
        self.saved_individuals.append(saved_individual)

    def generate_attacker(self, dataset):
        pass

    def get_hessian_vector_product(self, gradients, noises):
        pass


class NoiseGenerator(object):

    def __init__(self):
        pass


class Scorer(object):

    def __init__(self):
        pass
