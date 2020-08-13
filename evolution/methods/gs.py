"""
Name: NEAT evolved by Golden-Section Search

Function(s):
Reproduction by Golden-Section Search and Random Near Search.
"""

from ReverseEncodingTree.evolution.bean.genome import create_golden_section_new
from ReverseEncodingTree.evolution.methods import bi


class Reproduction(bi.Reproduction):

    def obtain_global_genome(self, matrix_1, matrix_2, saved_genomes, index):
        """
        obtain global genome based on the feather matrix in two genomes.

        :param matrix_1: one feature matrix.
        :param matrix_2: another feature matrix.
        :param saved_genomes: genomes are saved in this population before.
        :param index: genome index.

        :return: novel global (golden-section search) genome or not (cannot create due to min_distance).
        """
        center_genome = create_golden_section_new(matrix_1, matrix_2, self.genome_config, index)
        is_input = True
        for check_genome in saved_genomes:
            if center_genome.distance(check_genome, self.genome_config) < self.reproduction_config.min_distance:
                is_input = False

        if is_input:
            return center_genome

        return None
