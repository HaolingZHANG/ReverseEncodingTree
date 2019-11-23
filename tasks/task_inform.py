from enum import Enum

import gym
import neat

from evolution_process.bean import genome, species_set
from evolution_process.evolutor import FitDevice, FitProcess, TYPE_CORRECT, EVAL_TYPE
from evolution_process.methods import bi, tri, gs
from utils.operator import Operator


class METHOD_TYPE(Enum):
    FS = 0
    BI = 1
    GS = 2
    TRI = 3


class XOR(object):

    def __init__(self, method_type,
                 max_generation, display_results=False, checkpoint=-1, stdout=False):

        # 2-input XOR inputs and expected outputs.
        xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

        # load evolution process.
        fitter = FitDevice(FitProcess(init_fitness=4, eval_type=EVAL_TYPE.ManhattanDistance))
        fitter.set_dataset({"i": xor_inputs, "o": xor_outputs})

        # load configuration.
        config = None
        if method_type == METHOD_TYPE.FS:
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 "../configures/task/xor.fs")
            self.filename = "XOR.fs"
        elif method_type == METHOD_TYPE.BI:
            config = neat.Config(genome.GlobalGenome, bi.Reproduction,
                                 species_set.StrongSpeciesSet, neat.DefaultStagnation,
                                 "../configures/task/xor.bi")
            self.filename = "XOR.bi"
        elif method_type == METHOD_TYPE.GS:
            config = neat.Config(genome.GlobalGenome, gs.Reproduction,
                                 species_set.StrongSpeciesSet, neat.DefaultStagnation,
                                 "../configures/task/xor.gs")
            self.filename = "XOR.gs"
        elif method_type == METHOD_TYPE.TRI:
            config = neat.Config(genome.GlobalGenome, tri.Reproduction,
                                 species_set.StrongSpeciesSet, neat.DefaultStagnation,
                                 "../configures/task/xor.tri")
            self.filename = "XOR.tri"

        # initialize the NeuroEvolution
        self.operator = Operator(config=config, fitter=fitter,
                                 node_names={-1: 'A', -2: 'B', 0: 'A XOR B'},
                                 max_generation=max_generation, checkpoint=checkpoint, stdout=stdout,
                                 output_path="../output/")

        # set whether display results
        self.display_results = display_results

        self.max_generation = max_generation

    def run(self, times):
        generations = []
        for time in range(times):
            if times > 1:
                # print current times.
                print()
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print("procession time: " + str(time + 1))
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print()

            self.operator.obtain_winner()
            actual_generation, fit = self.operator.get_actual_generation()
            if not fit:
                time -= 1
                continue
            generations.append(actual_generation)

            # reset the hyper-parameters
            self.operator.reset()

        counts = [0 for _ in range(self.max_generation + 1)]
        for generation in generations:
            counts[generation] += 1

        if self.display_results:
            self.operator.display_genome(filename=self.filename)

        return generations, counts


class CartPole_v0(object):

    def __init__(self, method_type,
                 episode_steps, episode_generation,
                 max_generation, display_results=False, checkpoint=-1, stdout=False):

        # load evolution process.
        fitter = FitDevice(FitProcess())
        fitter.set_environment(environment=gym.make("CartPole-v0").unwrapped,
                               input_type=TYPE_CORRECT.List, output_type=TYPE_CORRECT.Value,
                               episode_steps=episode_steps, episode_generation=episode_generation)

        # load configuration.
        config = None
        if method_type == METHOD_TYPE.FS:
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 "../configures/task/cart-pole-v0.fs")
            self.filename = "CartPole-v0.fs"
        elif method_type == METHOD_TYPE.BI:
            config = neat.Config(genome.GlobalGenome, bi.Reproduction,
                                 species_set.StrongSpeciesSet, neat.DefaultStagnation,
                                 "../configures/task/cart-pole-v0.bi")
            self.filename = "CartPole-v0.bi"
        elif method_type == METHOD_TYPE.GS:
            config = neat.Config(genome.GlobalGenome, gs.Reproduction,
                                 species_set.StrongSpeciesSet, neat.DefaultStagnation,
                                 "../configures/task/cart-pole-v0.gs")
            self.filename = "CartPole-v0.gs"
        elif method_type == METHOD_TYPE.TRI:
            config = neat.Config(genome.GlobalGenome, tri.Reproduction,
                                 species_set.StrongSpeciesSet, neat.DefaultStagnation,
                                 "../configures/task/cart-pole-v0.tri")
            self.filename = "CartPole-v0.tri"

        # initialize the NeuroEvolution
        self.operator = Operator(config=config, fitter=fitter,
                                 node_names={-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', 0: 'act1', 1: 'act2'},
                                 max_generation=max_generation, checkpoint=checkpoint, stdout=stdout,
                                 output_path="../output/")

        # set whether display results
        self.display_results = display_results

        self.max_generation = max_generation

    def run(self, times):
        generations = []
        for time in range(times):
            if times > 1:
                # print current times.
                print()
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print("procession time: " + str(time + 1))
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print()

            self.operator.obtain_winner()
            actual_generation, fit = self.operator.get_actual_generation()
            if not fit:
                time -= 1
                continue
            generations.append(actual_generation)

            # reset the hyper-parameters
            self.operator.reset()
        counts = [0 for _ in range(self.max_generation + 1)]
        for generation in generations:
            counts[generation] += 1

        if self.display_results:
            self.operator.display_genome(filename=self.filename)

        return generations, counts


def save_distribution(counts, parent_path, task_name, method_type):
    """
    save the distribution of the generations.

    :param counts: counts of each generation in the requested times.
    :param parent_path: parent path for saving file.
    :param task_name: task name, like XOR.
    :param method_type: type of method in evolution process.
    """
    path = parent_path + task_name + "."
    if method_type == METHOD_TYPE.FS:
        path += "fs.csv"
    elif method_type == METHOD_TYPE.BI:
        path += "bi.csv"
    elif method_type == METHOD_TYPE.GS:
        path += "gs.csv"
    elif method_type == METHOD_TYPE.TRI:
        path += "tri.csv"

    with open(path, "w", encoding="utf-8") as save_file:
        for index, value in enumerate(counts):
            save_file.write(str(index) + ", " + str(value) + "\n")
