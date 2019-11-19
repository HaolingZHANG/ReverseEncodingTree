import gym
import neat

from evolution_process.evolutor import FitDevice, FitProcess
from utils.operator import Operator

environment = gym.make("CartPole-v0").unwrapped


def start_normal():
    # load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "../configures/example.cart-pole-v0")

    # load evolution process.
    fitter = FitDevice(FitProcess())
    fitter.set_environment(environment=environment, episode_steps=300, episode_generation=10)

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter,
                        node_names={-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', 0: 'act1', 1: 'act2'},
                        checkpoint=5, stdout=True, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    # display the winning genome.
    operator.display_genome(filename="CartPole-v0.normal")

    # evaluate the NeuroEvolution.
    operator.evaluation(environment=environment)


# noinspection SpellCheckingInspection
if __name__ == '__main__':
    start_normal()
