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
        self.genome_type = None
        self.last_global_rate = None
        self.add_rate = None
        self.last_speed = None
        self.stagnate_flag = False

    @classmethod
    def parse_config(cls, param_dict):
        """
        add init and min distance in config.

        :param param_dict: parameter dictionary.

        :return: config.
        """
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 2),
                                   ConfigParameter('init_distance', float, 5),
                                   ConfigParameter('min_distance', float, 0.2),
                                   ConfigParameter('search_count', int, 1),
                                   ConfigParameter('add_rate', float, 1.1)])

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
                    raise Exception("init_distance is too large for the whole landscape," +
                                    "please reduce init_distance or try again!")

            new_genomes[key] = genome
            self.ancestors[key] = tuple()
            self.add_rate = self.reproduction_config.add_rate

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
        current_genomes, current_speed = self.obtain_current(config, species, pop_size)

        # obtain topological genomes and near genomes.
        topo_genomes, near_genomes = self.obtain_new_network(pop_size, current_genomes)

        # add global genome by evolution speed to eliminate stagnate.
        global_genomes = self.insert_global_search(current_genomes + topo_genomes + near_genomes, current_speed,
                                                   len(near_genomes), pop_size, config.fitness_threshold / 100.0)

        if len(global_genomes) > 0:
            near_genomes = near_genomes[:-len(global_genomes)]

        # aggregate final population
        new_population = {}
        for index, genome in enumerate(current_genomes + topo_genomes + near_genomes + global_genomes):
            genome.key = index
            new_population[index] = genome

        return new_population

    def obtain_current(self, config, species, pop_size):
        """
        obtain current genotypical network and evolution speed.

        :param config: genome config.
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

        # calculate average adjusted fitness
        avg_adjusted_fitness = 0
        for genome in current_genomes:
            avg_adjusted_fitness += genome.fitness / pop_size
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # sort members in order of descending fitness.
        current_genomes.sort(reverse=True, key=lambda g: g.fitness)

        if len(current_genomes) > pop_size:
            current_genomes = current_genomes[:pop_size]

        return current_genomes, current_genomes[0].fitness / config.fitness_threshold

    def obtain_new_network(self, pop_size, current_genomes):
        """
        obtain new phenotypical network from population size and current genotypical network.

        :param pop_size: population size.
        :param current_genomes: current genotypical network.

        :return: center genomes and near genomes.
        """
        near_genomes = []
        center_genomes = []
        for index_1 in range(pop_size):
            genome_1 = current_genomes[index_1]
            for index_2 in range(pop_size):
                count = 0
                genome_2 = current_genomes[index_2]

                if genome_1.distance(genome_2, self.genome_config) > self.reproduction_config.min_distance:
                    # add near genome (limit search count)
                    while count < self.reproduction_config.search_count:
                        near_genome = create_near_new(genome_1, self.genome_config,
                                                      pop_size + len(near_genomes) + len(center_genomes))
                        is_input = True
                        for check_genome in current_genomes + near_genomes + center_genomes:
                            if near_genome.distance(check_genome, self.genome_config) \
                                    < self.reproduction_config.min_distance:
                                is_input = False
                        if is_input:
                            near_genomes.append(near_genome)
                            break
                        count += 1

                    # add center genome
                    center_genome = create_center_new(genome_1, genome_2, self.genome_config,
                                                      pop_size + len(near_genomes) + len(center_genomes))
                    is_input = True
                    for check_genome in current_genomes + near_genomes + center_genomes:
                        if center_genome.distance(check_genome, self.genome_config) \
                                < self.reproduction_config.min_distance:
                            is_input = False

                    if is_input and center_genome is not None:
                        center_genomes.append(center_genome)

        return center_genomes, near_genomes

    def insert_global_search(self, previous_genomes, current_speed, near_count, pop_size, change_threshold):
        global_genomes = []

        if self.last_global_rate is None:
            # record global rate and evolution speed.
            self.last_global_rate = pop_size / len(previous_genomes)
            self.last_speed = current_speed
        else:
            # add global genome by evolution speed to eliminate stagnate.
            if self.stagnate_flag and (current_speed - self.last_speed) > change_threshold:
                global_rate = pop_size / len(previous_genomes)
                self.stagnate_flag = False
            else:
                if current_speed == self.last_speed:
                    self.stagnate_flag = True

                global_rate = self.last_global_rate * self.add_rate / (2.0 * (current_speed / self.last_speed) - 1)

                if global_rate > 0.5:
                    global_rate = 0.5
                elif global_rate < pop_size / len(previous_genomes):
                    global_rate = pop_size / len(previous_genomes)

            print("Global count: " + str(int(near_count * global_rate)))
            for created_index in range(int(near_count * global_rate)):
                genome = self.genome_type(created_index + len(previous_genomes))
                for count in range(self.reproduction_config.search_count):
                    genome.configure_new(self.genome_config)
                    min_distance = float("inf")
                    for generated_genome in previous_genomes + global_genomes:
                        current_distance = genome.distance(generated_genome, self.genome_config)
                        if min_distance > current_distance:
                            min_distance = current_distance
                    if min_distance >= self.reproduction_config.init_distance:
                        global_genomes.append(genome)
                        break

            self.last_speed = current_speed
            self.last_global_rate = global_rate

        return global_genomes
