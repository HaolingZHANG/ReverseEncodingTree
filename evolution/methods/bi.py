"""
Name: NEAT evolved by Binary Search

Function(s):
Reproduction by Binary Search and Random Near Search.
"""
import copy

import math
import pandas
from sklearn.cluster import KMeans, SpectralClustering, Birch
from neat.reproduction import DefaultReproduction
from neat.config import DefaultClassConfig, ConfigParameter

from ReverseEncodingTree.evolution.bean.genome import create_near_new, create_center_new, distance_between_two_matrices


class Reproduction(DefaultReproduction):

    def __init__(self, config, reporters, stagnation):
        super().__init__(config, reporters, stagnation)
        self.best_genome = None
        self.genome_config = None
        self.genome_type = None
        self.global_rate = None

    @classmethod
    def parse_config(cls, param_dict):
        """
        add init and min distance in config.

        :param param_dict: parameter dictionary.

        :return: config.
        """
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('init_distance', float, 5),
                                   ConfigParameter('min_distance', float, 0.2),
                                   ConfigParameter('correlation_rate', float, -0.5),
                                   ConfigParameter('search_count', int, 1),
                                   ConfigParameter('cluster_method', str, "kmeans++")])

    def create_new(self, genome_type, genome_config, num_genomes):
        """
        create new genomes by type, config, and number.

        :param genome_type: genome type.
        :param genome_config: genome config.
        :param num_genomes: number of new genomes.

        :return: new genomes.
        """

        if genome_config.num_hidden + genome_config.num_inputs + genome_config.num_outputs > genome_config.max_node_num:
            raise Exception("config: max_node_num must larger than num_inputs + num_outputs + num_hidden")

        self.genome_config = genome_config
        self.genome_type = genome_type

        new_genomes = {}
        distance_matrix = [[float("inf") for _ in range(num_genomes - 1)] for _ in range(num_genomes - 1)]

        for created_index in range(num_genomes):
            key = next(self.genome_indexer)
            genome = genome_type(key)
            count = 0
            while True:
                genome.configure_new(genome_config)
                min_distance = float("inf")
                for check_index, new_genome in new_genomes.items():
                    current_distance = genome.distance(new_genome, genome_config)
                    distance_matrix[created_index - 1][check_index - 1] = current_distance
                    distance_matrix[check_index - 1][created_index - 1] = current_distance
                    if min_distance > current_distance:
                        min_distance = current_distance
                if min_distance >= self.reproduction_config.init_distance:
                    break

                count += 1

                if count > self.reproduction_config.search_count:
                    raise Exception("init_distance is too large for the whole landscape," +
                                    "please reduce init_distance or try again!")

            new_genomes[key] = genome
            self.ancestors[key] = tuple()

        return new_genomes

    def reproduce(self, config, species, pop_size, generation):
        """
        handles creation of genomes, either from scratch or by sexual or asexual reproduction from parents.

        :param config: genome config.
        :param species: genome species.
        :param pop_size: population size.
        :param generation: generation of population.

        :return: new population.
        """

        # obtain current genomes and current evolution speed.
        genome_clusters, cluster_centers = self.obtain_clusters(species, pop_size)

        # obtain topological genomes and near genomes.
        new_genomes = self.obtain_phenotypic_network(pop_size, genome_clusters, cluster_centers)

        # aggregate final population
        new_population = {}
        for index, genome in enumerate(new_genomes):
            genome.key = index
            new_population[index] = genome

        return new_population

    def obtain_clusters(self, species, pop_size):
        """
        obtain current genotypical network and evolution speed.

        :param species: genome species.
        :param pop_size: population size.

        :return:
        """
        # obtain all genomes from species.
        current_genomes = []
        for i, value in species.species.items():
            members = value.members
            for key, individual in members.items():
                current_genomes.append(individual)

        # sort members in order of descending fitness.
        current_genomes.sort(reverse=True, key=lambda g: g.fitness)

        # calculate speed list and avg adjusted fitness
        avg_adjusted_fitness = 0

        if len(current_genomes) > pop_size:
            feature_matrices = []
            for genome in current_genomes:
                feature_matrices.append([])
                for feature_slice in genome.feature_matrix:
                    feature_matrices[-1] += copy.deepcopy(feature_slice)

            # cluster the current network based on the size of population.
            labels, centers = self.cluster(feature_matrices, pop_size, len(current_genomes))

            genome_clusters = [[] for _ in range(pop_size)]
            for index, cluster_index in enumerate(labels):
                genome_clusters[cluster_index].append(current_genomes[index])

            for genome_cluster in genome_clusters:
                avg_adjusted_fitness += genome_cluster[0].fitness / pop_size

            self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

            return genome_clusters, centers
        else:
            genome_clusters = []
            for genome in current_genomes:
                genome_clusters.append([genome])
                avg_adjusted_fitness += genome.fitness / pop_size

            self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

            return genome_clusters, None

    def obtain_phenotypic_network(self, pop_size, genome_clusters, cluster_centers):
        """
        obtain new phenotypic network from population size and current phenotypic network.

        :param pop_size: population size.
        :param genome_clusters: current phenotypic network.
        :param cluster_centers: centers of cluster.

        :return: center genomes and near genomes.
        """
        if cluster_centers is not None:
            saved_genomes = []
            correlations = []

            # analyze the correlation between fitting degree and spatial position (negative correlation normally).
            for genome_cluster in genome_clusters:
                distances = [0]
                fitnesses = [genome_cluster[0].fitness]
                saved_genomes.append(genome_cluster[0])
                for index in range(1, len(genome_cluster)):
                    distances.append(genome_cluster[0].distance(genome_cluster[index], self.genome_config))
                    fitnesses.append(genome_cluster[index].fitness)

                if len(fitnesses) > 1:
                    correlations.append(round(pandas.Series(distances).corr(pandas.Series(fitnesses)), 2))
                else:
                    correlations.append(-1.00)

                for index in range(len(correlations)):
                    if math.isnan(correlations[index]):
                        correlations[index] = 0

            print("Correlations: " + str(correlations))

            new_genomes = []
            # construct the topology of the phenotypical network
            for index_1 in range(pop_size):
                cluster_1 = genome_clusters[index_1]
                for index_2 in range(index_1 + 1, pop_size):
                    cluster_2 = genome_clusters[index_2]

                    if distance_between_two_matrices(cluster_1[0].feature_matrix, cluster_2[0].feature_matrix) \
                            > self.reproduction_config.min_distance:

                        # If the two clusters both have highly correlations,
                        # it means that the current network of these two clusters has a better description of phenotype,
                        # and then evolution is carried out according to the original method.
                        if correlations[index_1] >= self.reproduction_config.correlation_rate \
                                and correlations[index_2] >= self.reproduction_config.correlation_rate:
                            topo_genome = self.obtain_global_genome(cluster_centers[index_1],
                                                                    cluster_centers[index_2],
                                                                    saved_genomes + new_genomes, -1)
                            if cluster_1[0].fitness > cluster_2[0].fitness:
                                near_genome = self.obtain_near_genome(cluster_1[0],
                                                                      saved_genomes + new_genomes + cluster_1, -1)
                            else:
                                near_genome = self.obtain_near_genome(cluster_2[0],
                                                                      saved_genomes + new_genomes + cluster_2, -1)

                            if near_genome is not None:
                                new_genomes.append(near_genome)
                            if topo_genome is not None:
                                new_genomes.append(topo_genome)

                        elif correlations[index_1] >= self.reproduction_config.correlation_rate > correlations[index_2]:
                            if cluster_1[0].fitness > cluster_2[0].fitness:
                                near_genome_1 = self.obtain_near_genome(cluster_1[0],
                                                                        saved_genomes + new_genomes + cluster_1, -1)
                                near_genome_2 = self.obtain_near_genome(cluster_2[0],
                                                                        saved_genomes + new_genomes + cluster_2, -1)
                            else:
                                near_genome_1 = self.obtain_near_genome(cluster_2[0],
                                                                        saved_genomes + new_genomes + cluster_2, -1)
                                near_genome_2 = self.obtain_near_genome(cluster_2[0],
                                                                        saved_genomes + new_genomes + cluster_2, -1)

                            if near_genome_1 is not None:
                                new_genomes.append(near_genome_1)
                            if near_genome_2 is not None:
                                new_genomes.append(near_genome_2)

                        elif correlations[index_2] >= self.reproduction_config.correlation_rate > correlations[index_1]:
                            if cluster_1[0].fitness > cluster_2[0].fitness:
                                near_genome_1 = self.obtain_near_genome(cluster_1[0],
                                                                        saved_genomes + new_genomes + cluster_1, -1)
                                near_genome_2 = self.obtain_near_genome(cluster_1[0],
                                                                        saved_genomes + new_genomes + cluster_1, -1)

                            else:
                                near_genome_1 = self.obtain_near_genome(cluster_1[0],
                                                                        saved_genomes + new_genomes + cluster_1, -1)
                                near_genome_2 = self.obtain_near_genome(cluster_2[0],
                                                                        saved_genomes + new_genomes + cluster_2, -1)

                            if near_genome_1 is not None:
                                new_genomes.append(near_genome_1)
                            if near_genome_2 is not None:
                                new_genomes.append(near_genome_2)
                        else:
                            near_genome_1 = self.obtain_near_genome(cluster_1[0],
                                                                    saved_genomes + new_genomes + cluster_1, -1)
                            near_genome_2 = self.obtain_near_genome(cluster_2[0],
                                                                    saved_genomes + new_genomes + cluster_2, -1)
                            if near_genome_1 is not None:
                                new_genomes.append(near_genome_1)
                            if near_genome_2 is not None:
                                new_genomes.append(near_genome_2)

            new_genomes += saved_genomes
        else:
            # create the initial topology network (binary search & near search).
            new_genomes = []
            for genome_cluster in genome_clusters:
                new_genomes.append(genome_cluster[0])

            for index_1 in range(pop_size):
                genome_1 = new_genomes[index_1]
                for index_2 in range(index_1 + 1, pop_size):
                    genome_2 = new_genomes[index_2]

                    if genome_1.distance(genome_2, self.genome_config) > self.reproduction_config.min_distance:
                        # add near genome (limit search count)
                        near_genome = self.obtain_near_genome(genome_1, new_genomes, -1)
                        # add center genome
                        topo_genome = self.obtain_global_genome(genome_1.feature_matrix, genome_2.feature_matrix,
                                                                new_genomes, -1)
                        if near_genome is not None:
                            new_genomes.append(near_genome)
                        if topo_genome is not None:
                            new_genomes.append(topo_genome)

        return new_genomes

    def cluster(self, feature_matrices, pop_size, iteration):
        """
        cluster the current network based on the size of population using Cluster Method.

        :param feature_matrices: set of feature matrix (one dimensio).
        :param pop_size: population size.
        :param iteration: maximum iteration.

        :return: labels and cluster centers.
        """
        centers = []
        if self.reproduction_config.cluster_method == "kmeans++":
            method = KMeans(n_clusters=pop_size, max_iter=iteration)
        elif self.reproduction_config.cluster_method == "spectral":
            method = SpectralClustering(n_clusters=pop_size)
        elif self.reproduction_config.cluster_method == "birch":
            method = Birch(n_clusters=pop_size)
        else:
            method = KMeans(n_clusters=pop_size, max_iter=iteration, init="random")

        method.fit(feature_matrices)
        for cluster_center in method.cluster_centers_:
            feature_matrix = []
            for index in range(self.genome_config.max_node_num):
                feature_matrix.append(list(cluster_center[index * self.genome_config.max_node_num:
                                                          (index + 1) * self.genome_config.max_node_num + 1]))
            centers.append(feature_matrix)

        return method.labels_, centers

    def obtain_global_genome(self, matrix_1, matrix_2, saved_genomes, index):
        """
        obtain global genome based on the feather matrix in two genomes.

        :param matrix_1: one feature matrix.
        :param matrix_2: another feature matrix.
        :param saved_genomes: genomes are saved in this population before.
        :param index: genome index.

        :return: novel global (binary search) genome or not (cannot create due to min_distance).
        """
        center_genome = create_center_new(matrix_1, matrix_2, self.genome_config, index)
        is_input = True
        for check_genome in saved_genomes:
            if center_genome.distance(check_genome, self.genome_config) < self.reproduction_config.min_distance:
                is_input = False

        if is_input:
            return center_genome

        return None

    def obtain_near_genome(self, parent_genome, saved_genomes, index):
        """
        obtain near genome by NEAT.

        :param parent_genome: parent genome.
        :param saved_genomes: genomes are saved in this population before.
        :param index: genome index.

        :return: novel near genome or not (cannot create due to min_distance).
        """
        count = 0
        while count < self.reproduction_config.search_count:
            near_genome = create_near_new(parent_genome, self.genome_config, index)
            is_input = True
            for check_genome in saved_genomes:
                if near_genome.distance(check_genome, self.genome_config) < self.reproduction_config.min_distance:
                    is_input = False
            if is_input:
                return near_genome

            count += 1

        return None
