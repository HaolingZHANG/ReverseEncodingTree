"""
Name: NEAT evoluted by Golden-Section Search

Function(s):
Reproduction by Golden-Section Search and Random Near Search.
"""
from evolution_process.bean.genome import create_near_new, create_golden_section_new
from evolution_process.methods import bi


class Reproduction(bi.Reproduction):

    def reproduce(self, config, species, pop_size, generation):
        """
        handles creation of genomes, either from scratch or by sexual or asexual reproduction from parents.

        :param config: genome config.
        :param species: genome species.
        :param pop_size: population size.
        :param generation: generation of population.

        :return: new population.
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
        current_genomes.sort(reverse=True, key=lambda x: x.fitness)

        if len(current_genomes) > pop_size:
            current_genomes = current_genomes[:pop_size]

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
                                    < self.reproduction_config.min_distance:
                                is_input = False

                        if is_input:
                            new_genomes.append(near_genome)
                            break

                        count += 1

                    # add center genome
                    center_genome = create_golden_section_new(genome_1, genome_2, self.genome_config,
                                                              pop_size + len(new_genomes))
                    is_input = True
                    for check_genome in current_genomes + new_genomes:
                        if center_genome.distance(check_genome, self.genome_config) \
                                < self.reproduction_config.min_distance:
                            is_input = False

                    if is_input and center_genome is not None:
                        new_genomes.append(center_genome)

        # aggregate final population
        new_population = {}
        for index, genome in enumerate(current_genomes + new_genomes):
            genome.key = index
            new_population[index] = genome

        return new_population
