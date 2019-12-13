from enum import Enum

import gym
import neat

from evolution_process.bean import genome, species_set
from evolution_process.bean.phenotyper import create_drosophila_melanogaster, screen
from evolution_process.evolutor import FitDevice, FitProcess, TYPE_CORRECT, EVAL_TYPE
from evolution_process.methods import bi, tri, gs
from utils.operator import Operator


class METHOD_TYPE(Enum):
    FS = 0
    BI = 1
    GSS = 2
    TRI = 3


class LOGIC_TYPE(Enum):
    NAND = 1
    NOR = 2
    IMPLY = 3
    XOR = 4


class BIO_TYPE(Enum):
    DROSOPHILA_MELANOGASTER = 1
    FLOWER = 2


class GAME_TYPE(Enum):
    CartPole_v0 = 0


class Logic(object):

    def __init__(self, method_type, logic_type,
                 max_generation, display_results=False, checkpoint=-1, stdout=False):

        data_inputs = None
        data_outputs = None

        if logic_type == LOGIC_TYPE.NAND:
            data_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
            data_outputs = [(1.0,), (1.0,), (1.0,), (0.0,)]
            self.filename = "nand."
        elif logic_type == LOGIC_TYPE.NOR:
            data_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
            data_outputs = [(0.0,), (0.0,), (0.0,), (1.0,)]
            self.filename = "nor."
        elif logic_type == LOGIC_TYPE.IMPLY:
            data_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
            data_outputs = [(1.0,), (1.0,), (0.0,), (1.0,)]
            self.filename = "imply."
        elif logic_type == LOGIC_TYPE.XOR:
            data_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
            data_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]
            self.filename = "xor."

        # load evolution process.
        fitter = FitDevice(FitProcess(init_fitness=4, eval_type=EVAL_TYPE.ManhattanDistance))
        fitter.set_dataset({"i": data_inputs, "o": data_outputs})

        # load configuration.
        config = None
        if method_type == METHOD_TYPE.FS:
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 "../configures/task/logic.fs")
            self.filename += "fs"
        elif method_type == METHOD_TYPE.BI:
            config = neat.Config(genome.GlobalGenome, bi.Reproduction,
                                 species_set.StrongSpeciesSet, neat.DefaultStagnation,
                                 "../configures/task/logic.bi")
            self.filename += "bi"
        elif method_type == METHOD_TYPE.GSS:
            config = neat.Config(genome.GlobalGenome, gs.Reproduction,
                                 species_set.StrongSpeciesSet, neat.DefaultStagnation,
                                 "../configures/task/logic.gss")
            self.filename += "gss"
        elif method_type == METHOD_TYPE.TRI:
            config = neat.Config(genome.GlobalGenome, tri.Reproduction,
                                 species_set.StrongSpeciesSet, neat.DefaultStagnation,
                                 "../configures/task/logic.tri")
            self.filename += "tri"

        # initialize the NeuroEvolution
        self.operator = Operator(config=config, fitter=fitter,
                                 node_names={-1: 'A', -2: 'B', 0: 'A operate B'},
                                 max_generation=max_generation, checkpoint=checkpoint, stdout=stdout,
                                 output_path="../output/")

        # set whether display results
        self.display_results = display_results

        self.max_generation = max_generation

    def run(self, times):
        generations = []
        time = 0
        while True:
            # try:
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
                    # reset the hyper-parameters
                    self.operator.reset()
                    continue
                generations.append(actual_generation)

                time += 1
                if time >= times:
                    break

                # reset the hyper-parameters
                self.operator.reset()
            # except Exception or ValueError:
            #     print("something error.")
            #     self.operator.reset()

        counts = [0 for _ in range(self.max_generation + 1)]
        for generation in generations:
            counts[generation] += 1

        if self.display_results:
            self.operator.display_genome(filename=self.filename)

        return generations, counts


class Biology(object):

    def __init__(self, method_type, bio_type,
                 max_generation, selection_ratio=1,
                 display_results=False, checkpoint=-1, stdout=False):

        data_inputs = None
        data_outputs = None

        if bio_type == BIO_TYPE.DROSOPHILA_MELANOGASTER:
            data_inputs, data_outputs = create_drosophila_melanogaster()
            self.filename = "drosophila_melanogaster."
        # elif bio_type == BIO_TYPE.FLOWER:
        #     data_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        #     data_outputs = [(0.0,), (0.0,), (0.0,), (1.0,)]
        #     self.filename = "flower."

        # selection if requested.
        if selection_ratio < 1:
            data_inputs, data_outputs = screen(data_inputs, data_outputs, selection_ratio)

        # load evolution process.
        fitter = FitDevice(FitProcess(init_fitness=0, eval_type=EVAL_TYPE.ManhattanDistance))
        fitter.set_dataset({"i": data_inputs, "o": data_outputs})

        # load configuration.
        config = None
        if method_type == METHOD_TYPE.FS:
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 "../configures/task/bio.fs")
            self.filename += "fs"
        elif method_type == METHOD_TYPE.BI:
            config = neat.Config(genome.GlobalGenome, bi.Reproduction,
                                 species_set.StrongSpeciesSet, neat.DefaultStagnation,
                                 "../configures/task/bio.bi")
            self.filename += "bi"
        elif method_type == METHOD_TYPE.GSS:
            config = neat.Config(genome.GlobalGenome, gs.Reproduction,
                                 species_set.StrongSpeciesSet, neat.DefaultStagnation,
                                 "../configures/task/bio.gss")
            self.filename += "gs"
        elif method_type == METHOD_TYPE.TRI:
            config = neat.Config(genome.GlobalGenome, tri.Reproduction,
                                 species_set.StrongSpeciesSet, neat.DefaultStagnation,
                                 "../configures/task/bio.tri")
            self.filename += "tri"

        # initialize the NeuroEvolution
        self.operator = Operator(config=config, fitter=fitter,
                                 node_names={-1: '1 gene a', -2: '1 gene b', -3: '2 gene a', -4: '2 gene b',
                                             0: 'ab', 1: 'ab', 2: 'Ab', 3: 'AB'},
                                 max_generation=max_generation, checkpoint=checkpoint, stdout=stdout,
                                 output_path="../output/")

        # set whether display results
        self.display_results = display_results

        self.max_generation = max_generation

    def run(self, times):
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
                if not fit:
                    # reset the hyper-parameters
                    self.operator.reset()
                    continue
                generations.append(actual_generation)

                time += 1
                if time >= times:
                    break

                # reset the hyper-parameters
                self.operator.reset()
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

        game_environment = None
        if game_type == GAME_TYPE.CartPole_v0:
            game_environment = gym.make("CartPole-v0").unwrapped

        # load evolution process.
        fitter = FitDevice(FitProcess())
        fitter.set_environment(environment=game_environment,
                               input_type=TYPE_CORRECT.List, output_type=TYPE_CORRECT.Value,
                               episode_steps=episode_steps, episode_generation=episode_generation,
                               attacker=attacker, noise_level=noise_level)

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
        elif method_type == METHOD_TYPE.GSS:
            config = neat.Config(genome.GlobalGenome, gs.Reproduction,
                                 species_set.StrongSpeciesSet, neat.DefaultStagnation,
                                 "../configures/task/cart-pole-v0.gss")
            self.filename = "CartPole-v0.gss"
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
                if not fit:
                    # reset the hyper-parameters
                    self.operator.reset()
                    continue
                generations.append(actual_generation)

                time += 1
                if time >= times:
                    break

                # reset the hyper-parameters
                self.operator.reset()
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
    if method_type == METHOD_TYPE.FS:
        path += "fs.csv"
    elif method_type == METHOD_TYPE.BI:
        path += "bi.csv"
    elif method_type == METHOD_TYPE.GSS:
        path += "gss.csv"
    elif method_type == METHOD_TYPE.TRI:
        path += "tri.csv"

    with open(path, "w", encoding="utf-8") as save_file:
        for index, value in enumerate(counts):
            save_file.write(str(index) + ", " + str(value) + "\n")
