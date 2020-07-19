from __future__ import print_function

import copy
import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy


def plot_statistics(statistics, y_log=False, show=False, parent_path="", filename=None):
    """
    Plot the population's average and best fitness.

    :param statistics: statistics of NeuroEvolution.
    :param y_log: whether y-axis needs log.
    :param show: whether the view is showable.
    :param parent_path: parent path of output files.
    :param filename: file name of the output.
    """
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = numpy.array(statistics.get_fitness_mean())
    stdev_fitness = numpy.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if y_log:
        plt.gca().set_yscale('symlog')

    if filename is None:
        plt.savefig(parent_path + "fitness.svg")
    else:
        plt.savefig(parent_path + filename + "_fitness.svg")

    if show:
        plt.show()

    plt.close()


def plot_species(statistics, show=False, parent_path="", filename=None):
    """
    Visualize speciation throughout evolution.

    :param statistics: statistics of NeuroEvolution.
    :param show: whether the view is showable.
    :param parent_path: parent path of output files.
    :param filename: file name of the output.
    """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = numpy.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    if filename is None:
        plt.savefig(parent_path + "speciation.svg")
    else:
        plt.savefig(parent_path + filename + "_speciation.svg")

    if show:
        plt.show()

    plt.close()


def draw_network(config, genome, show=False, parent_path="", filename="best_network", node_names=None,
                 show_disabled=True, prune_unused=False, node_colors=None, file_format='svg'):
    """
    Receive a genome and draw a neural network with arbitrary topology.

    :param config: configures of NEAT.
    :param genome: genome of network.
    :param show: whether the view is showable.
    :param parent_path: parent path of output files.
    :param filename: filename of the genome.
    :param node_names: node information for display the obtained network.
    :param show_disabled: show disabled.
    :param prune_unused: prune unused.
    :param node_colors: colors of node.
    :param file_format: format of the saved file.
    """
    # attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {'shape': 'circle', 'fontsize': '9', 'width': '0.2', 'height': '0.2'}

    dot = graphviz.Digraph(format=file_format, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input_value, output_value = cg.key
            a = node_names.get(input_value, str(input_value))
            b = node_names.get(output_value, str(output_value))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(parent_path + filename, view=show)

    return dot
