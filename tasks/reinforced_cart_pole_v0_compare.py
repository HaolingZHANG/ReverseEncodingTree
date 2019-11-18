import gym
import neat

from evolution_process.bean import genome
from evolution_process.evolutor import FitDevice, FitProcess
from evolution_process.methods import bi, tri
from example import reinforced_cart_pole_v0
from utils.operator import Operator

environment = gym.make("CartPole-v0").unwrapped


def start_normal():
    reinforced_cart_pole_v0.start_normal()


def start_bi():
    # load configuration.
    config = neat.Config(genome.GlobalGenome, bi.Reproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "../configures/task.bi.cart-pole-v0")

    # load evolution process.
    fitter = FitDevice(FitProcess())
    fitter.set_environment(environment=environment, episode_steps=300, episode_generation=10)

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter,
                        node_names={-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', 0: 'act1', 1: 'act2'},
                        generations=1, checkpoint=6,
                        stdout=True, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    # display the winning genome.
    operator.display_genome()


def start_tri():
    # load configuration.
    config = neat.Config(genome.GlobalGenome, tri.Reproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "../configures/task.bi.cart-pole-v0")

    # load evolution process.
    fitter = FitDevice(FitProcess())
    fitter.set_environment(environment=environment, episode_steps=300, episode_generation=10)

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter,
                        node_names={-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', 0: 'act1', 1: 'act2'},
                        generations=1, checkpoint=7,
                        stdout=True, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    # display the winning genome.
    operator.display_genome()


if __name__ == '__main__':
    start_normal()
    start_bi()
    start_tri()
