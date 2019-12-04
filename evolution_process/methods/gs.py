"""
Name: NEAT evoluted by Golden-Section Search

Function(s):
Reproduction by Golden-Section Search and Random Near Search.
"""

from evolution_process.bean.genome import create_near_new, create_golden_section_new
from evolution_process.methods import bi


class Reproduction(bi.Reproduction):

    def obtain_new_network(self, pop_size, current_genomes):
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
                    center_genome = create_golden_section_new(genome_1, genome_2, self.genome_config,
                                                              pop_size + len(near_genomes) + len(center_genomes))
                    is_input = True
                    for check_genome in current_genomes + near_genomes + center_genomes:
                        if center_genome.distance(check_genome, self.genome_config) \
                                < self.reproduction_config.min_distance:
                            is_input = False

                    if is_input and center_genome is not None:
                        center_genomes.append(center_genome)

        return center_genomes, near_genomes
