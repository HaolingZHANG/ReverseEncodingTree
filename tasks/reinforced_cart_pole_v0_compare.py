import gym
import neat

from evolution_process.bean import genome, species_set
from evolution_process.evolutor import FitDevice, FitProcess
from evolution_process.methods import bi, tri
from example import reinforced_cart_pole_v0
from utils.operator import Operator

environment = gym.make("CartPole-v0").unwrapped


def start_normal(need_play=True):
    """
    run CartPole-v0 by GS-NEAT.

    :param need_play: if needs play after train.

    :return: the operator.
    """
    return reinforced_cart_pole_v0.start_normal(need_play)


def start_bi(max_generation, need_play=True):
    """
    run CartPole-v0 by GS-NEAT.

    :param need_play: if needs play after train.

    :return: the operator.
    """
    # load configuration.
    config = neat.Config(genome.GlobalGenome, bi.Reproduction,
                         species_set.StrongSpeciesSet, neat.DefaultStagnation,
                         "../configures/task.bi.cart-pole-v0")

    # load evolution process.
    fitter = FitDevice(FitProcess())
    fitter.set_environment(environment=environment, episode_steps=300, episode_generation=10)

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter,
                        node_names={-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', 0: 'act1', 1: 'act2'},
                        generations=max_generation, checkpoint=6, stdout=False, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    # display the winning genome.
    if need_play:
        operator.display_genome(filename="CartPole-v0.bi")

    # return operator if need other operations
    return operator


def start_gs(need_play=True):
    """
    run CartPole-v0 by GS-NEAT.

    :param need_play: if needs play after train.

    :return: the operator.
    """

    # load configuration.
    config = neat.Config(genome.GlobalGenome, tri.Reproduction,
                         species_set.StrongSpeciesSet, neat.DefaultStagnation,
                         "../configures/task.gs.cart-pole-v0")

    # load evolution process.
    fitter = FitDevice(FitProcess())
    fitter.set_environment(environment=environment, episode_steps=300, episode_generation=10)

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter,
                        node_names={-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', 0: 'act1', 1: 'act2'},
                        checkpoint=7, stdout=True, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    # display the winning genome.
    if need_play:
        operator.display_genome(filename="CartPole-v0.gs")

    # return operator if need other operations
    return operator


def start_tri(need_play=True):
    """
    run CartPole-v0 by Tri-NEAT.

    :param need_play: if needs play after train.

    :return: the operator.
    """

    # load configuration.
    config = neat.Config(genome.GlobalGenome, tri.Reproduction,
                         species_set.StrongSpeciesSet, neat.DefaultStagnation,
                         "../configures/task.tri.cart-pole-v0")

    # load evolution process.
    fitter = FitDevice(FitProcess())
    fitter.set_environment(environment=environment, episode_steps=300, episode_generation=10)

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter,
                        node_names={-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', 0: 'act1', 1: 'act2'},
                        checkpoint=8, stdout=False, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    # display the winning genome.
    if need_play:
        operator.display_genome(filename="CartPole-v0.tri")

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