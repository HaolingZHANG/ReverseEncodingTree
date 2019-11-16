import copy
from itertools import count

import math
import neat
from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from neat.config import ConfigParameter
from neat.genes import DefaultNodeGene, DefaultConnectionGene
from neat.genome import DefaultGenomeConfig
from six import iteritems, itervalues, iterkeys


def create_center_new(genome_1, genome_2, config, key):
    """
    create a new genome at the midpoint of two genomes.

    :param genome_1: one genome.
    :param genome_2: another genome.
    :param config: genome config.
    :param key: key of the new genome.

    :return:
    """
    if hasattr(genome_1, 'feature_matrix') and hasattr(genome_2, 'feature_matrix') \
            and len(genome_1.feather_matrix) == len(genome_2.feather_matrix) \
            and len(genome_1.feather_matrix[0]) == len(genome_2.feather_matrix[0]):
        length = len(genome_1.feather_matrix)
        new_feature_matrix = [[0 for _ in range(length)] for _ in range(length)]
        for row in range(length):
            for col in range(length):
                new_feature_matrix[row][col] = (genome_1.feather_matrix[row][col] +
                                                genome_2.feather_matrix[row][col]) / 2
        new_genome = config.genome_type(key)
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


class GlobalGenomeConfig(neat.DefaultGenomeConfig):

    def __init__(self, params):
        """
        initialize config by params, add ConfigParameter('max_num', int)
        """
        # Create full set of available activation functions.
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

    def get_new_node_key(self, node_dict):
        if self.node_indexer is None:
            self.node_indexer = count(max(list(iterkeys(node_dict))) + 1)

        new_id = next(self.node_indexer)

        assert new_id not in node_dict

        return new_id

    def check_structural_mutation_surer(self):
        if self.structural_mutation_surer == 'true':
            return True
        elif self.structural_mutation_surer == 'false':
            return False
        elif self.structural_mutation_surer == 'default':
            return self.single_structural_mutation
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)


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
        # print("config.activation_default = " + str(config.activation_default))
        # print("config.aggregation_default = " + str(config.aggregation_default))
        super().configure_new(config)
        # print(self)
        self.set_feature_matrix(config)
        # print(self.feature_matrix)
        # exit(121)

    def feature_matrix_new(self, feature_matrix, config):
        """
        create new genome by feature matrix.

        :param feature_matrix: obtained feature matrix.
        :param config: genome config
        """
        self.feature_matrix = feature_matrix

        # set nodes by feature matrix
        for index in range(config.num_hidden):
            node = DefaultNodeGene(index)
            node.bias = feature_matrix[index + config.num_inputs][0]
            node.response = config.response_init_mean
            node.activation = config.activation_default
            node.aggregation = config.aggregation_default
            self.nodes[index] = node

        # set connections by feature matrix
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
