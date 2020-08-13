import gym

from neat import config, genome, reproduction, species, stagnation

from ReverseEncodingTree.evolution.evolutor import FitDevice, FitProcess, TypeCorrect
from ReverseEncodingTree.utils.operator import Operator

has_environment = False
for environment_space in gym.envs.registry.all():
    if "LunarLander-v2" in environment_space.id:
        has_environment = True
        break

if not has_environment:
    raise Exception("no environment named LunarLander-v2.")

environment = gym.make("LunarLander-v2")

if __name__ == '__main__':
    task_config = config.Config(genome.DefaultGenome, reproduction.DefaultReproduction,
                                species.DefaultSpeciesSet, stagnation.DefaultStagnation,
                                "../configures/example/lunar-lander-v2")

    # load evolution process.
    fitter = FitDevice(FitProcess())

    fitter.set_environment(environment=environment, episode_steps=300, episode_generation=5,
                           input_type=TypeCorrect.List, output_type=TypeCorrect.Value)

    # initialize the NeuroEvolution
    operator = Operator(config=task_config, fitter=fitter,
                        node_names={-1: '1', -2: '2', -3: '3', -4: '4',
                                    -5: '5', -6: '6', -7: '7', -8: '8',
                                    0: 'fire engine'},
                        max_generation=1000, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    # display the winning genome.
    operator.display_genome(filename="example.LunarLander-v2.fs")

    # evaluate the NeuroEvolution.
    operator.evaluation(environment=environment)
