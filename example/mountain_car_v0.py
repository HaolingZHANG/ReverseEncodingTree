import gym
import neat

from evolution_process.evolutor import FitDevice, FitProcess, TYPE_CORRECT
from utils.operator import Operator

has_environment = False
for environment_space in gym.envs.registry.all():
    if "MountainCar-v0" in environment_space.id:
        has_environment = True
        break

if not has_environment:
    raise Exception("no environment named MountainCar-v0.")

environment = gym.make("MountainCar-v0")

if __name__ == '__main__':
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "../configures/example/mountain-car-v0.fs")

    # load evolution process.
    fitter = FitDevice(FitProcess())

    fitter.set_environment(environment=environment, episode_steps=200, episode_generation=5,
                           input_type=TYPE_CORRECT.List, output_type=TYPE_CORRECT.Value)

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter,
                        node_names={-1: 'feature 1', -2: 'feature 2',
                                    0: 'action 1', 1: 'action 2', 2: 'action 3'},
                        max_generation=100, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    # display the winning genome.
    operator.display_genome(filename="example.MountainCar-v0.fs")

    # evaluate the NeuroEvolution.
    operator.evaluation(environment=environment)
