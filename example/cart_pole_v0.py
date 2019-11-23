import gym
import neat

from evolution_process.evolutor import FitDevice, FitProcess, TYPE_CORRECT
from utils.operator import Operator

environment = gym.make("CartPole-v0").unwrapped


if __name__ == '__main__':
    # load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "../configures/example/cart-pole-v0")

    # load evolution process.
    fitter = FitDevice(FitProcess(), input_type=TYPE_CORRECT.List, output_type=TYPE_CORRECT.Value)
    fitter.set_environment(environment=environment, episode_steps=300, episode_generation=10)

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter,
                        node_names={-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', 0: 'act1', 1: 'act2'},
                        max_generation=500, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    # display the winning genome.
    operator.display_genome(filename="example.CartPole-v0.fs")
    # evaluate the NeuroEvolution.
    operator.evaluation(environment=environment)