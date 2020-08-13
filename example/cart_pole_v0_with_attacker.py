import gym

from neat import config, genome, reproduction, species, stagnation

from ReverseEncodingTree.evolution.bean.attacker import CartPole_v0_Attacker, AttackType
from ReverseEncodingTree.evolution.evolutor import FitDevice, FitProcess, TypeCorrect
from ReverseEncodingTree.utils.operator import Operator

has_environment = False
for environment_space in gym.envs.registry.all():
    if "CartPole-v0" in environment_space.id:
        has_environment = True
        break

if not has_environment:
    raise Exception("no environment named CartPole-v0.")

environment = gym.make("CartPole-v0").unwrapped
environment.reset()

if __name__ == '__main__':
    # load configuration.
    task_config = config.Config(genome.DefaultGenome, reproduction.DefaultReproduction,
                                species.DefaultSpeciesSet, stagnation.DefaultStagnation,
                                "../configures/example/cart-pole-v0")

    # load evolution process.
    fitter = FitDevice(FitProcess())
    attacker = CartPole_v0_Attacker(attack_type=AttackType.Normal)
    fitter.set_environment(environment=environment, episode_steps=300, episode_generation=10,
                           input_type=TypeCorrect.List, output_type=TypeCorrect.Value,
                           attacker=attacker, noise_level=0.5)

    # initialize the NeuroEvolution
    operator = Operator(config=task_config, fitter=fitter,
                        node_names={-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', 0: 'act1', 1: 'act2'},
                        max_generation=500, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    # display the winning genome.
    operator.display_genome(filename="example.CartPole-v0.fs")

    # evaluate the NeuroEvolution.
    operator.evaluation(environment=environment,
                        input_type=TypeCorrect.List, output_type=TypeCorrect.Value)
