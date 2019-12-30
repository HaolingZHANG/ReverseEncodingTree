import itertools
import math

# Dominant Homozygote: e.g. AA
DHo = 2
# Dominant Heterozygote: e.g. Aa
DHe = 1
# Negative Homozygote: e.g. aa
NHo = 0

# Hybridization ratio: (negative, dominant), the position is the combination of gene type.
normal_hybridize = [[(1.00, 0.00), (0.50, 0.50), (0.00, 1.00)],
                    [(0.50, 0.50), (0.25, 0.75), (0.00, 1.00)],
                    [(0.00, 1.00), (0.00, 1.00), (0.00, 1.00)]]


def create_drosophila_melanogaster():
    """
    create genes from Drosophila Melanogaster (Normal).

    Drosophila Melanogaster: eye (red [dominant], white [negative]), body (gray [dominant], black [negative]).
    input: eye color in individual 1, body color in individual 1, eye color in individual 2, body color in individual 2.
    output: white eye + black body, white eye + gray body, red eye + black body, red eye + gray body.

    ref: Adams, M. D., Celniker, S. E., Holt, R. A., Evans, C. A., Gocayne, J. D., Amanatides, P. G., ... & George, R. A. (2000). The genome sequence of Drosophila melanogaster. Science, 287(5461), 2185-2195.

    :return: data_inputs, data_outputs.
    """
    return create_phenotypes(gene_count=2)


def create_flower():
    """
    Additive Gene Effects

    ref: Bateson, W., & Saunders, E. R. (1910). Reports to the Evolution Committee of the Royal Society: ReportsI-V, 1902-09. Royal Society.

    :return:
    """
    gene_count = 2
    data_inputs, data_outputs = [], []

    gene_types = list(itertools.product([NHo, DHe, DHo], repeat=gene_count))
    for gene_1 in gene_types:
        for gene_2 in gene_types:
            # calculate the genotypical situation (input)
            data_input = [0 for _ in range(gene_count * 2)]
            for gene_index in range(gene_count):
                data_input[gene_index] = gene_1[gene_index]
                data_input[gene_index + gene_count] = gene_2[gene_index]

            # calculate the phenotypical distribution (output)
            distribution = []


def create_phenotypes(gene_count, death_types=None, influences=None):
    """
    create the phenotypical distribution.

    :param gene_count:
    :param death_types:
    :param influences:

    :return:
    """
    data_inputs, data_outputs = [], []

    gene_types = list(itertools.product([NHo, DHe, DHo], repeat=gene_count))
    for gene_1 in gene_types:
        for gene_2 in gene_types:
            # delete the death types
            if death_types is not None and (gene_1 in death_types or gene_2 in death_types):
                continue

            # calculate the genotypical situation (input)
            data_input = [0 for _ in range(gene_count * 2)]
            for gene_index in range(gene_count):
                data_input[gene_index] = gene_1[gene_index]
                data_input[gene_index + gene_count] = gene_2[gene_index]

            # calculate the phenotypical distribution (output)
            distribution = []
            for target_type in range(2 ** gene_count):
                detailed_type = list(map(int, list(str(bin(target_type))[2:].zfill(gene_count))))
                count = math.pow(2, gene_count) * 4
                for gene_index in range(gene_count):
                    count *= normal_hybridize[gene_1[gene_index]][gene_2[gene_index]][detailed_type[gene_index]]
                distribution.append(count)

            data_inputs.append(tuple(data_input))
            data_outputs.append(tuple(distribution))

    return data_inputs, data_outputs


def screen(inputs, outputs, selection_rate=1):
    selected_inputs, selected_outputs = [], []
    interval = math.ceil(1 / selection_rate)
    for index, (data_input, data_output) in enumerate(zip(inputs, outputs)):
        if index % interval == 0:
            selected_inputs.append(data_input)
            selected_outputs.append(data_output)

    return selected_inputs, selected_outputs


def select(inputs, outputs, selections):
    selected_inputs, selected_outputs = [], []
    for index, (data_input, data_output) in enumerate(zip(inputs, outputs)):
        if index in selections:
            selected_inputs.append(data_input)
            selected_outputs.append(data_output)

    return selected_inputs, selected_outputs
