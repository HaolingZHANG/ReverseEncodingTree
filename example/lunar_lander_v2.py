import gym
import neat

from evolution_process.evolutor import FitDevice, FitProcess, TYPE_CORRECT
from utils.operator import Operator

has_environment = False
for environment_space in gym.envs.registry.all():
    if "LunarLander-v2" in environment_space.id:
        has_environment = True
        break

if not has_environment:
    raise Exception("no environment named LunarLander-v2.")

environment = gym.make("LunarLander-v2")

if __name__ == '__main__':
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "../configures/example/lunar-lander-v2")

    # load evolution process.
    fitter = FitDevice(FitProcess())

    fitter.set_environment(environment=environment, episode_steps=1000, episode_generation=5,
                           input_type=TYPE_CORRECT.List, output_type=TYPE_CORRECT.Value)

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter,
                        node_names={-1: 'cos(theta)', -2: 'sin(theta)', -3: 'theta dot', 0: 'joint effort'},
                        max_generation=10000, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    # display the winning genome.
    operator.display_genome(filename="example.LunarLander-v2.fs")

    # evaluate the NeuroEvolution.
    operator.evaluation(environment=environment)
