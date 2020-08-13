from neat import config, genome, reproduction, species, stagnation

from ReverseEncodingTree.evolution.evolutor import EvalType, FitDevice, FitProcess
from ReverseEncodingTree.utils.operator import Operator

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

if __name__ == '__main__':
    # load configuration.
    task_config = config.Config(genome.DefaultGenome, reproduction.DefaultReproduction,
                                species.DefaultSpeciesSet, stagnation.DefaultStagnation,
                                "../configures/example/xor")

    # load evolution process.
    fitter = FitDevice(FitProcess(init_fitness=4, eval_type=EvalType.ManhattanDistance))
    fitter.set_dataset({"i": xor_inputs, "o": xor_outputs})

    # initialize the NeuroEvolution
    operator = Operator(config=task_config, fitter=fitter, node_names={-1: 'A', -2: 'B', 0: 'A XOR B'},
                        max_generation=500, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    # display the winning genome.
    operator.display_genome(filename="example.xor.fs")
