import gym
import neat

from evolution_process.evolutor import FitDevice, FitProcess, TYPE_CORRECT
from utils.operator import Operator

has_environment = False
for environment_space in gym.envs.registry.all():
    if "Taxi-v2" in environment_space.id:
        has_environment = True
        break

if not has_environment:
    raise Exception("no environment named Taxi-v2.")

environment = gym.make("Taxi-v2").unwrapped

if __name__ == '__main__':
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "../configures/example/taxi-v2")

    # load evolution process.
    fitter = FitDevice(FitProcess())

    fitter.set_environment(environment=environment, episode_steps=300, episode_generation=10,
                           input_type=TYPE_CORRECT.List, output_type=TYPE_CORRECT.Value)

    # initialize the NeuroEvolution
    operator = Operator(config=config, fitter=fitter,
                        node_names={-1: 'state', 0: 'action'},
                        output_path="../output/", stdout=True)

    # obtain the winning genome.
    operator.obtain_winner()

    # display the winning genome.
    operator.display_genome(filename="example.Taxi-v2.fs")

    # # evaluate the NeuroEvolution.
    # operator.evaluation(environment=environment,
    #                     input_type=TYPE_CORRECT.List, output_type=TYPE_CORRECT.Value)
