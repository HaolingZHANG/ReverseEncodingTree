"""
Name: NEAT evoluted by Binary Search

Function(s):
Reproduction by Binary Search and Random Near Search.
"""

from neat import DefaultReproduction
from neat.config import DefaultClassConfig, ConfigParameter

from evolution_process.bean.genome import create_near_new, create_center_new


class Reproduction(DefaultReproduction):

    def __init__(self, config, reporters, stagnation):
        super().__init__(config, reporters, stagnation)
        self.genome_config = None

    @classmethod
    def parse_config(cls, param_dict):
        """
        add init and min distance in config.

        :param param_dict: parameter dictory.

        :return: config.
        """
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 2),
                                   ConfigParameter('init_distance', float, 5),
                                   ConfigParameter('min_distance', float, 0.2),
                                   ConfigParameter('search_count', int, 1)])

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

        new_genomes = {}
        distance_matrix = [[float("inf") for _ in range(num_genomes - 1)] for _ in range(num_genomes - 1)]

        for created_index in range(num_genomes):
            key = next(self.genome_indexer)
            genome = genome_type(key)
            count = 0
            while True:
                genome.configure_new(genome_config)
                min_distance = float("inf")
                for check_index in range(1, created_index):
                    current_distance = genome.distance(new_genomes[check_index], genome_config)
                    distance_matrix[created_index - 1][check_index - 1] = current_distance
                    distance_matrix[check_index - 1][created_index - 1] = current_distance
                    if min_distance > current_distance:
                        min_distance = current_distance
                if min_distance >= self.reproduction_config.init_distance:
                    break

                count += 1

                if count > self.reproduction_config.search_count:
                    raise Exception('init_distance is too large for the whole landscape.')

            new_genomes[key] = genome
            self.ancestors[key] = tuple()

        # print("distance:")
        # for i in distance_matrix:
        #     print(i)

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
        current_genomes = []
        for key, value in species.species.items():
            current_genomes.append(value.members.get(key))

        # sort members in order of descending fitness.
        current_genomes.sort(reverse=True, key=lambda x: x.fitness)

        if len(current_genomes) > pop_size:
            current_genomes = current_genomes[:pop_size]

        # calculate average adjusted fitness
        avg_adjusted_fitness = 0
        for genome in current_genomes:
            avg_adjusted_fitness += genome.fitness / pop_size
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        new_genomes = []
        for index_1 in range(pop_size):
            genome_1 = current_genomes[index_1]
            for index_2 in range(pop_size):
                count = 0
                genome_2 = current_genomes[index_2]

                if genome_1.distance(genome_2, self.genome_config) > self.reproduction_config.min_distance:
                    # add near genome (limit search count)
                    while count < self.reproduction_config.search_count:
                        near_genome = create_near_new(genome_1, self.genome_config, pop_size + len(new_genomes))
                        is_input = True
                        for check_genome in current_genomes + new_genomes:
                            if near_genome.distance(check_genome, self.genome_config) \
                                    > self.reproduction_config.min_distance:
                                is_input = False

                        if is_input:
                            new_genomes.append(near_genome)
                            break

                        count += 1

                    # add center genome
                    center_genome = create_center_new(genome_1, genome_2, self.genome_config,
                                                      pop_size + len(new_genomes))
                    is_input = True
                    for check_genome in current_genomes + new_genomes:
                        if center_genome.distance(check_genome, self.genome_config) \
                                > self.reproduction_config.min_distance:
                            is_input = False

                    if is_input:
                        new_genomes.append(center_genome)

        print("pop_size = " + str(pop_size))
        print("new_genomes = " + str(len(new_genomes)))

        new_population = {}

        for index, genome in enumerate(current_genomes + new_genomes):
            new_population[index] = genome

        # TODO some bugs in self.species
        # Traceback (most recent call last):
        #     File "E:/Bi-NEAT/Code/Bi-NEAT/tasks/supervised_xor_compare.py", line 32, in <module>
        #         operator.obtain_winner()
        #     File "E:\Bi-NEAT\Code\Bi-NEAT\utils\operator.py", line 52, in obtain_winner
        #         self._winner = self._population.run(self._fitter.genomes_fitness, self._generations)
        #     File "D:\Professional\Python3.7.3\lib\site-packages\neat\population.py", line 127, in run
        #         self.species.speciate(self.config, self.population, self.generation)
        #     File "D:\Professional\Python3.7.3\lib\site-packages\neat\species.py", line 96, in speciate
        #         unspeciated.remove(new_rid)
        #     KeyError: 4

        return new_population
