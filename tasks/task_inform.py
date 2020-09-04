from enum import Enum

import gym
from neat import config, genome, reproduction, species, stagnation

import ReverseEncodingTree.evolution.bean.genome as autogenome
import ReverseEncodingTree.evolution.bean.species_set as autospecies

from ReverseEncodingTree.evolution.evolutor import TypeCorrect, EvalType
from ReverseEncodingTree.evolution.evolutor import FitDevice, FitProcess
from ReverseEncodingTree.evolution.methods import bi, gs
from ReverseEncodingTree.utils.operator import Operator


class MethodType(Enum):
    N = 0
    FS = 1
    BI = 2
    GS = 3


class LogicType(Enum):
    NAND = 1
    NOR = 2
    IMPLY = 3
    XOR = 4


class GameType(Enum):
    CartPole_v0 = 0
    LunarLander_v2 = 1


class Logic(object):

    def __init__(self, method_type, logic_type,
                 max_generation, display_results=False, checkpoint=-1, stdout=False):
        """
        initialize the logical task.

        :param method_type: the evolution strategy, FS-NEAT, Bi-NEAT, or GS-NEAT.
        :param logic_type: the task type, IMPLY, NAND, NOR, or XOR.
        :param max_generation: maximum generation of the evolution strategy,
                               if the generation exceeds the maximum, it will be terminated.
        :param display_results: whether result visualization is required.
        :param checkpoint: check the statistics point.
        :param stdout: Whether outputting the genome information in the process is required.
        """

        data_inputs = None
        data_outputs = None

        if logic_type == LogicType.NAND:
            data_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
            data_outputs = [(1.0,), (1.0,), (1.0,), (0.0,)]
            self.filename = "nand."
        elif logic_type == LogicType.NOR:
            data_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
            data_outputs = [(0.0,), (0.0,), (0.0,), (1.0,)]
            self.filename = "nor."
        elif logic_type == LogicType.IMPLY:
            data_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
            data_outputs = [(1.0,), (1.0,), (0.0,), (1.0,)]
            self.filename = "imply."
        elif logic_type == LogicType.XOR:
            data_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
            data_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]
            self.filename = "xor."

        # load evolution process.
        fitter = FitDevice(FitProcess(init_fitness=4, eval_type=EvalType.ManhattanDistance))
        fitter.set_dataset({"i": data_inputs, "o": data_outputs})

        # load configuration.
        task_config = None
        if method_type == MethodType.N:
            task_config = config.Config(genome.DefaultGenome, reproduction.DefaultReproduction,
                                        species.DefaultSpeciesSet, stagnation.DefaultStagnation,
                                        "../configures/task/logic.n")
            self.filename += "fs"
        elif method_type == MethodType.FS:
            task_config = config.Config(genome.DefaultGenome, reproduction.DefaultReproduction,
                                        species.DefaultSpeciesSet, stagnation.DefaultStagnation,
                                        "../configures/task/logic.fs")
            self.filename += "fs"
        elif method_type == MethodType.BI:
            task_config = config.Config(autogenome.GlobalGenome, bi.Reproduction,
                                        autospecies.StrongSpeciesSet, stagnation.DefaultStagnation,
                                        "../configures/task/logic.bi")
            self.filename += "bi"
        elif method_type == MethodType.GS:
            task_config = config.Config(autogenome.GlobalGenome, gs.Reproduction,
                                        autospecies.StrongSpeciesSet, stagnation.DefaultStagnation,
                                        "../configures/task/logic.gs")
            self.filename += "gs"

        # initialize the NeuroEvolution
        self.operator = Operator(config=task_config, fitter=fitter,
                                 node_names={-1: 'A', -2: 'B', 0: 'A operate B'},
                                 max_generation=max_generation, checkpoint_value=checkpoint, stdout=stdout,
                                 output_path="../output/")

        # set whether display results
        self.display_results = display_results

        self.max_generation = max_generation

    def run(self, times):
        """
        multi-run task.

        :param times: running times.

        :return: results of generation and counts of each generation in the requested times.
        """
        generations = []
        time = 0
        while True:
            try:
                if times > 1:
                    # print current times.
                    print()
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    print("procession time: " + str(time + 1))
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    print()

                self.operator.obtain_winner()
                actual_generation, fit = self.operator.get_actual_generation()

                time += 1
                if time >= times:
                    break

                # reset the hyper-parameters
                self.operator.reset()

                if not fit:
                    continue

                generations.append(actual_generation)
            except Exception or ValueError:
                print("something error.")
                self.operator.reset()

        counts = [0 for _ in range(self.max_generation + 1)]
        for generation in generations:
            counts[generation] += 1

        if self.display_results:
            self.operator.display_genome(filename=self.filename)

        return generations, counts


class Game(object):

    def __init__(self, method_type, game_type,
                 episode_steps, episode_generation, max_generation,
                 attacker=None, noise_level=-1,
                 display_results=False, checkpoint=-1, stdout=False):
        """
        initialize the game task.

        :param method_type: the evolution strategy, FS-NEAT, Bi-NEAT, or GS-NEAT.
        :param game_type: the task type, CartPole_v0 or LunarLander_v2.
        :param episode_steps: step parameter of game task
        :param episode_generation:  generation parameter of game task.
        :param max_generation: maximum generation of the evolution strategy,
                               if the generation exceeds the maximum, it will be terminated.
        :param attacker: noise attacker, see evolution/bean/attacker.py.
        :param noise_level: noise level.
        :param display_results: whether result visualization is required.
        :param checkpoint: check the statistics point.
        :param stdout: Whether outputting the genome information in the process is required.
        """

        game_environment = None
        if game_type == GameType.CartPole_v0:
            game_environment = gym.make("CartPole-v0").unwrapped
            self.filename = "cart-pole-v0."
            self.node_name = {-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', 0: 'act1', 1: 'act2'}
        elif game_type == GameType.LunarLander_v2:
            game_environment = gym.make("LunarLander-v2")
            self.filename = "lunar-lander-v2."
            self.node_name = {-1: '1', -2: '2', -3: '3', -4: '4', -5: '5', -6: '6', -7: '7', -8: '8', 0: 'fire engine'}

        fitter = FitDevice(FitProcess())
        fitter.set_environment(environment=game_environment,
                               input_type=TypeCorrect.List, output_type=TypeCorrect.Value,
                               episode_steps=episode_steps, episode_generation=episode_generation,
                               attacker=attacker, noise_level=noise_level)
        # load configuration.
        task_config = None
        if method_type == MethodType.N:
            self.filename += "n"
            task_config = config.Config(genome.DefaultGenome, reproduction.DefaultReproduction,
                                        species.DefaultSpeciesSet, stagnation.DefaultStagnation,
                                        "../configures/task/" + self.filename)
        elif method_type == MethodType.FS:
            self.filename += "fs"
            task_config = config.Config(genome.DefaultGenome, reproduction.DefaultReproduction,
                                        species.DefaultSpeciesSet, stagnation.DefaultStagnation,
                                        "../configures/task/" + self.filename)
        elif method_type == MethodType.BI:
            self.filename += "bi"
            task_config = config.Config(autogenome.GlobalGenome, bi.Reproduction,
                                        autospecies.StrongSpeciesSet, stagnation.DefaultStagnation,
                                        "../configures/task/" + self.filename)
        elif method_type == MethodType.GS:
            self.filename += "gs"
            task_config = config.Config(autogenome.GlobalGenome, gs.Reproduction,
                                        autospecies.StrongSpeciesSet, stagnation.DefaultStagnation,
                                        "../configures/task/" + self.filename)

        # initialize the NeuroEvolution
        self.operator = Operator(config=task_config, fitter=fitter, node_names=self.node_name,
                                 max_generation=max_generation, checkpoint_value=checkpoint, stdout=stdout,
                                 output_path="../output/")

        # set whether display results
        self.display_results = display_results

        self.max_generation = max_generation

    def run(self, times):
        """
        multi-run task.

        :param times: running times.

        :return: results of generation and counts of each generation in the requested times.
        """
        generations = []
        time = 0
        while True:
            try:
                if times > 1:
                    # print current times.
                    print()
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    print("procession time: " + str(time + 1))
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    print()

                self.operator.obtain_winner()
                actual_generation, fit = self.operator.get_actual_generation()

                time += 1
                if time >= times:
                    break

                # reset the hyper-parameters
                self.operator.reset()

                if not fit:
                    continue

                generations.append(actual_generation)
            except Exception or ValueError:
                print("something error.")
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
    if method_type == MethodType.N:
        path += "n.csv"
    elif method_type == MethodType.FS:
        path += "fs.csv"
    elif method_type == MethodType.BI:
        path += "bi.csv"
    elif method_type == MethodType.GS:
        path += "gs.csv"

    with open(path, "w", encoding="utf-8") as save_file:
        for index, value in enumerate(counts):
            save_file.write(str(index) + ", " + str(value) + "\n")
