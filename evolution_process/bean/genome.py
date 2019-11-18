import copy
import random

import math
import neat
from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from neat.config import ConfigParameter
from neat.genes import DefaultNodeGene, DefaultConnectionGene
from neat.genome import DefaultGenomeConfig
from six import iteritems, itervalues


def create_center_new(genome_1, genome_2, config, key):
    """
    create a new genome at the midpoint of two genomes.

    :param genome_1: one genome.
    :param genome_2: another genome.
    :param config: genome config.
    :param key: key of the new genome.

    :return: center genome.
    """
    if hasattr(genome_1, 'feature_matrix') and hasattr(genome_2, 'feature_matrix'):
        length = len(genome_1.feature_matrix)
        new_feature_matrix = [[0 for _ in range(length)] for _ in range(length)]
        for row in range(length):
            for col in range(length):
                new_feature_matrix[row][col] = (genome_1.feature_matrix[row][col] +
                                                genome_2.feature_matrix[row][col]) / 2
        new_genome = GlobalGenome(key)
        new_genome.feature_matrix_new(new_feature_matrix, config)
        return new_genome
    return None


def create_near_new(genome, config, key):
    """
    create a new genome near the old genome.

    :param genome: original genome.
    :param config: genome config.
    :param key: key of new genome.

    :return: the new genome.
    """
    new_genome = copy.deepcopy(genome)
    new_genome.key = key
    new_genome.mutate(config)
    new_genome.set_feature_matrix(config)

    return new_genome


# noinspection PyMissingConstructor
class GlobalGenomeConfig(DefaultGenomeConfig):

    def __init__(self, params):
        """
        initialize config by params, add ConfigParameter('max_num', int)
        """
        # create full set of available activation functions.
        self.num_inputs = 0
        self.num_outputs = 0
        self.single_structural_mutation = None
        self.activation_defs = ActivationFunctionSet()
        # ditto for aggregation functions - name difference for backward compatibility
        self.aggregation_function_defs = AggregationFunctionSet()
        self.aggregation_defs = self.aggregation_function_defs

        self._params = [ConfigParameter('num_inputs', int),
                        ConfigParameter('num_outputs', int),
                        ConfigParameter('num_hidden', int),
                        ConfigParameter('max_node_num', int),
                        ConfigParameter('feed_forward', bool),
                        ConfigParameter('compatibility_disjoint_coefficient', float),
                        ConfigParameter('compatibility_weight_coefficient', float),
                        ConfigParameter('conn_add_prob', float),
                        ConfigParameter('conn_delete_prob', float),
                        ConfigParameter('node_add_prob', float),
                        ConfigParameter('node_delete_prob', float),
                        ConfigParameter('single_structural_mutation', bool, 'false'),
                        ConfigParameter('structural_mutation_surer', str, 'default'),
                        ConfigParameter('initial_connection', str, 'unconnected')]

        # Gather configuration data from the gene classes.
        self.node_gene_type = params['node_gene_type']
        self._params += self.node_gene_type.get_config_params()
        self.connection_gene_type = params['connection_gene_type']
        self._params += self.connection_gene_type.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

        self.connection_fraction = None

        # Verify that initial connection type is valid.
        # pylint: disable=access-member-before-definition
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in DefaultGenomeConfig.allowed_connectivity

        # Verify structural_mutation_surer is valid.
        if self.structural_mutation_surer.lower() in ['1', 'yes', 'true', 'on']:
            self.structural_mutation_surer = 'true'
        elif self.structural_mutation_surer.lower() in ['0', 'no', 'false', 'off']:
            self.structural_mutation_surer = 'false'
        elif self.structural_mutation_surer.lower() == 'default':
            self.structural_mutation_surer = 'default'
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)

        self.node_indexer = None


class GlobalGenome(neat.DefaultGenome):

    @classmethod
    def parse_config(cls, param_dict):
        super().parse_config(param_dict)
        return GlobalGenomeConfig(param_dict)

    def __init__(self, key):
        super().__init__(key)
        self.feature_matrix = None

    def configure_new(self, config):
        """
        create new genome by configure, and then create the feature matrix.

        :param config: genome config.
        """
        # create node genes for the output pins.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key)

        # add hidden nodes if requested.
        if config.num_hidden > 0:
            for node_key in range(len(config.output_keys), config.num_hidden + len(config.output_keys)):
                node = self.create_node(config, node_key)
                self.nodes[node_key] = node

        # add connections with global random.
        for input_id, output_id in self.compute_connections(config):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection
        # add feature matrix
        self.set_feature_matrix(config)

    @staticmethod
    def compute_connections(config):
        start = len(config.output_keys)
        stop = config.num_hidden + len(config.output_keys)
        hidden_keys = random.sample([index for index in range(start, stop)], random.randint(0, config.num_hidden))

        connections = []

        if len(hidden_keys) == 0:
            for input_id in config.input_keys:
                for output_id in config.output_keys:
                    if random.randint(0, 1):
                        connections.append((input_id, output_id))

            if len(connections) == 0:
                input_id = random.sample(config.input_keys, 1)
                for output_id in config.output_keys:
                    connections.append((input_id[0], output_id))
        else:
            chosen_keys = set()

            # from input and hidden nodes to hidden nodes.
            for index in range(len(hidden_keys)):
                before_keys = config.input_keys + hidden_keys[:index]
                for in_degree in random.sample(before_keys, random.randint(1, len(before_keys))):
                    connections.append((in_degree, hidden_keys[index]))
                    chosen_keys.add(in_degree)
                    chosen_keys.add(hidden_keys[index])

            # from input and hidden nodes to output nodes.
            chosen_keys = list(chosen_keys)
            for output_id in config.output_keys:
                for in_degree in random.sample(chosen_keys, random.randint(1, len(chosen_keys))):
                    connections.append((in_degree, output_id))

        connections = sorted(connections)

        return connections

    def feature_matrix_new(self, feature_matrix, config):
        """
        create new genome by feature matrix.

        :param feature_matrix: obtained feature matrix.
        :param config: genome config
        """
        self.feature_matrix = feature_matrix

        # create node genes for the output pins.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key)

        # add hidden nodes by feature matrix if requested.
        for node_key in range(config.num_hidden):
            node = DefaultNodeGene(node_key)
            node.bias = feature_matrix[node_key + config.num_inputs][0]
            node.response = config.response_init_mean
            node.activation = config.activation_default
            node.aggregation = config.aggregation_default
            self.nodes[node_key] = node

        # set connections by feature matrix.
        for in_index in range(len(feature_matrix)):
            for out_index in range(1, len(feature_matrix[in_index])):
                if feature_matrix[in_index][out_index] > 0:
                    connection = DefaultConnectionGene((in_index, out_index - 1))
                    connection.weight = feature_matrix[in_index][out_index]
                    connection.enabled = config.enabled_default
                    self.connections[connection.key] = connection

    def set_feature_matrix(self, config):
        """
        set the feature matrix for this genome.

        :param config: genome config.
        """
        # bia + weight
        self.feature_matrix = [[0 for _ in range(config.max_node_num + 1)] for _ in range(config.max_node_num)]

        # position mapping of feature matrix
        mapping = {}
        for index in range(config.num_inputs):
            mapping[index - config.num_inputs] = index

        # add node bias
        index = config.num_inputs
        for node_key, node_gene in iteritems(self.nodes):
            try:
                self.feature_matrix[index][0] = node_gene.bias
                mapping[node_key] = index
                index += 1
            except IndexError:
                print('index = ' + str(index))

        # add connect weight
        for connect_gene in itervalues(self.connections):
            row = mapping.get(connect_gene.key[0])
            col = mapping.get(connect_gene.key[1]) + 1
            self.feature_matrix[row][col] = connect_gene.weight

    def distance(self, other, config):
        """
        obtain distance by two feature matrix.

        :param other: another genome.
        :param config: genome config.

        :return: distance of two genomes.
        """
        if other.feature_matrix is None:
            other.set_feature_matrix(config)

        distance = 0
        for row in range(len(self.feature_matrix)):
            for col in range(len(self.feature_matrix)):
                distance += math.pow(self.feature_matrix[row][col] - other.feature_matrix[row][col], 2)

        return math.sqrt(distance)

    def mutate_add_node(self, config):
        """
        mutate add node when current hidden node (when node number less than the node range).

        :param config:genome config.
        """
        if config.max_node_num > len(self.nodes):
            super().mutate_add_node(config)

    def __str__(self):
        s = super().__str__()
        s += "\nFeature Matrix:"
        for row in self.feature_matrix:
            s += "\n\t" + str(row)

        return s
