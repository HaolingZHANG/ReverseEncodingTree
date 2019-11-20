import neat

from evolution_process.bean import genome, species_set
from evolution_process.evolutor import EVAL_TYPE, FitDevice, FitProcess
from evolution_process.methods import bi, tri, gs
from example import supervised_xor
from utils.operator import Operator


# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def start_normal():
    supervised_xor.start_normal()


def start_bi():
    # load configuration.
    config = neat.Config(genome.GlobalGenome, bi.Reproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "../configures/task.bi.xor")

    # load evolution process.
    fitter = FitDevice(FitProcess(init_fitness=4, eval_type=EVAL_TYPE.ManhattanDistance))
    fitter.set_dataset({"i": xor_inputs, "o": xor_outputs})

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter, node_names={-1: 'A', -2: 'B', 0: 'A XOR B'},
                        checkpoint=51, stdout=True, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    # display the winning genome.
    operator.display_genome(filename="xor.bi")


def start_gs():
    # load configuration.
    config = neat.Config(genome.GlobalGenome, gs.Reproduction,
                         species_set.StrongSpeciesSet, neat.DefaultStagnation,
                         "../configures/task.gs.xor")

    # load evolution process.
    fitter = FitDevice(FitProcess(init_fitness=4, eval_type=EVAL_TYPE.ManhattanDistance))
    fitter.set_dataset({"i": xor_inputs, "o": xor_outputs})

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter, node_names={-1: 'A', -2: 'B', 0: 'A XOR B'},
                        checkpoint=52, stdout=True, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    # display the winning genome.
    operator.display_genome(filename="xor.gs")


def start_tri():
    # load configuration.
    config = neat.Config(genome.GlobalGenome, tri.Reproduction,
                         species_set.StrongSpeciesSet, neat.DefaultStagnation,
                         "../configures/task.tri.xor")

    # load evolution process.
    fitter = FitDevice(FitProcess(init_fitness=4, eval_type=EVAL_TYPE.ManhattanDistance))
    fitter.set_dataset({"i": xor_inputs, "o": xor_outputs})

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter, node_names={-1: 'A', -2: 'B', 0: 'A XOR B'},
                        checkpoint=53, stdout=True, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    # display the winning genome.
    operator.display_genome(filename="xor.tri")


if __name__ == '__main__':
    # start_normal()
    # start_bi()
    start_gs()
    # start_tri()
