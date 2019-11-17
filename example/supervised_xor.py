import neat

from evolution_process.evolutor import EVAL_TYPE, FitDevice, FitProcess
from utils.operator import Operator


# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def start_normal():
    # load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "../configures/example.xor")

    # load evolution process.
    fitter = FitDevice(FitProcess(init_fitness=4, eval_type=EVAL_TYPE.ManhattanDistance))
    fitter.set_dataset({"i": xor_inputs, "o": xor_outputs})

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter,
                        node_names={-1: 'A', -2: 'B', 0: 'A XOR B'},
                        generations=300, checkpoint=50,
                        stdout=True, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    # display the winning genome.
    operator.display_genome()


if __name__ == '__main__':
    start_normal()
