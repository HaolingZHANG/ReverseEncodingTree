import gym
import neat

from evolution_process.evolutor import FitDevice, FitProcess, TYPE_CORRECT
from utils.operator import Operator

environment = gym.make("Pendulum-v0").unwrapped


if __name__ == '__main__':
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "../configures/example/pendulum-v0")

    # load evolution process.
    fitter = FitDevice(FitProcess(), input_type=TYPE_CORRECT.List, output_type=TYPE_CORRECT.List)
    fitter.set_environment(environment=environment, episode_steps=50, episode_generation=2)

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter,
                        node_names={-1: 'cos(theta)', -2: 'sin(theta)', -3: 'theta dot', 0: 'joint effort'},
                        max_generation=500, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    # display the winning genome.
    operator.display_genome(filename="example.Pendulum-v0.fs")
    # evaluate the NeuroEvolution.
    operator.evaluation(environment=environment)
