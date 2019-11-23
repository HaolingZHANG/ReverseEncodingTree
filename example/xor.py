import neat

from evolution_process.evolutor import EVAL_TYPE, FitDevice, FitProcess, TYPE_CORRECT
from utils.operator import Operator


# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def start_fs(max_generation, need_play=False):
    # load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "../configures/example/xor")

    # load evolution process.
    fitter = FitDevice(FitProcess(init_fitness=4, eval_type=EVAL_TYPE.ManhattanDistance))
    fitter.set_dataset({"i": xor_inputs, "o": xor_outputs})

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter, node_names={-1: 'A', -2: 'B', 0: 'A XOR B'},
                        max_generation=max_generation, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    if need_play:
        # display the winning genome.
        operator.display_genome(filename="example.xor.fs")

    # return operator if need other operations
    return operator.get_actual_generation()


def run(times=1):
    generations = []
    for i in range(times):
        if times == 1:
            actual_generation, fit = start_fs(max_generation=500, need_play=True)
            if not fit:
                i -= 1
        else:
            o = start_fs(max_generation=500, need_play=False)
            g, s = o.get_actual_generation()
            if s:
                generations.append(g)
            else:
                i -= 1
            print()
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("procession: " + str(i + 1))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print()

    if times > 1:
        print("generations:")
        print(generations)

        max_value = max(generations)
        counts = [0 for _ in range(max_value + 1)]
        for generation in generations:
            counts[generation] += 1

        print("counts:")
        print(counts)


if __name__ == '__main__':
    run(1)