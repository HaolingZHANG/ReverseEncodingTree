"""
Name: NEAT evoluted by Golden-Section Search

Function(s):
Reproduction by Golden-Section Search and Random Near Search.
"""

from evolution_process.bean.genome import create_golden_section_new
from evolution_process.methods import bi


class Reproduction(bi.Reproduction):

    def obtain_topological_genome(self, matrix_1, matrix_2, saved_genomes, index):
        center_genome = create_golden_section_new(matrix_1, matrix_2, self.genome_config, index)
        is_input = True
        for check_genome in saved_genomes:
            if center_genome.distance(check_genome, self.genome_config) < self.reproduction_config.min_distance:
                is_input = False

        if is_input:
            return center_genome

        return None
