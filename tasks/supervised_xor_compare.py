import neat

from evolution_process.bean import genome, species_set
from evolution_process.evolutor import EVAL_TYPE, FitDevice, FitProcess
from evolution_process.methods import bi, tri, gs
from example import supervised_xor
from utils.operator import Operator


# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def start_normal(max_generation, need_play=False):
    return supervised_xor.start_normal(max_generation, need_play)


def start_bi(max_generation, need_play=False):
    # load configuration.
    config = neat.Config(genome.GlobalGenome, bi.Reproduction,
                         species_set.StrongSpeciesSet, neat.DefaultStagnation,
                         "../configures/task.bi.xor")

    # load evolution process.
    fitter = FitDevice(FitProcess(init_fitness=4, eval_type=EVAL_TYPE.ManhattanDistance))
    fitter.set_dataset({"i": xor_inputs, "o": xor_outputs})

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter, node_names={-1: 'A', -2: 'B', 0: 'A XOR B'},
                        generations=max_generation, checkpoint=51, stdout=False, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    if need_play:
        # display the winning genome.
        operator.display_genome(filename="xor.bi")

    # return operator if need other operations
    return operator


def start_gs(max_generation, need_play=False):
    # load configuration.
    config = neat.Config(genome.GlobalGenome, gs.Reproduction,
                         species_set.StrongSpeciesSet, neat.DefaultStagnation,
                         "../configures/task.gs.xor")

    # load evolution process.
    fitter = FitDevice(FitProcess(init_fitness=4, eval_type=EVAL_TYPE.ManhattanDistance))
    fitter.set_dataset({"i": xor_inputs, "o": xor_outputs})

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter, node_names={-1: 'A', -2: 'B', 0: 'A XOR B'},
                        generations=max_generation, checkpoint=52, stdout=True, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    if need_play:
        # display the winning genome.
        operator.display_genome(filename="xor.gs")


def start_tri(max_generation, need_play=False):
    # load configuration.
    config = neat.Config(genome.GlobalGenome, tri.Reproduction,
                         species_set.StrongSpeciesSet, neat.DefaultStagnation,
                         "../configures/task.tri.xor")

    # load evolution process.
    fitter = FitDevice(FitProcess(init_fitness=4, eval_type=EVAL_TYPE.ManhattanDistance))
    fitter.set_dataset({"i": xor_inputs, "o": xor_outputs})

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter, node_names={-1: 'A', -2: 'B', 0: 'A XOR B'},
                        generations=max_generation, checkpoint=53, stdout=True, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    if need_play:
        # display the winning genome.
        operator.display_genome(filename="xor.tri")

    # return operator if need other operations
    return operator


if __name__ == '__main__':
    generations = []
    s_count = 0
    for i in range(1000):
        try:
            o = start_bi(max_generation=500, need_play=False)
            g, s = o.get_actual_generation()
            if s:
                generations.append(g)
            else:
                s_count += 1
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("procession: " + str(i + 1))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        except Exception:
            i -= 1

    print("generations:")
    print(generations)

    max_value = max(generations)
    counts = [0 for i in range(max_value + 1)]
    for generation in generations:
        counts[generation] += 1

    print("counts:")
    print(counts)
    print("obtain failed:")
    print(s_count)
