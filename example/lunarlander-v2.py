import gym
import neat

from evolution_process.evolutor import FitDevice, FitProcess
from utils.operator import Operator

# !conda install box2d
# !python -m pip install pyvirtualdisplay



is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

environment = gym.make("LunarLander-v2").unwrapped
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

# agent = Agent(state_size=8, action_size=4, seed=0)

def start_normal(max_generation, need_play=True):
    """
    run CartPole-v0 by normal-NEAT.

    :param max_generation: maximum generation, over return - 1.
    :param need_play: if needs play after train.

    :return: the operator.
    """
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
                        generations=max_generation, checkpoint=5, stdout=False, output_path="../output/")

    # obtain the winning genome.
    operator.obtain_winner()

    if need_play:
        # display the winning genome.
        operator.display_genome(filename="CartPole-v0.normal")
        # evaluate the NeuroEvolution.
        operator.evaluation(environment=environment)

    # return operator if need other operations
    return operator


# noinspection SpellCheckingInspection
if __name__ == '__main__':
    generations = []
    s_count = 0
    for i in range(1000):
        o = start_normal(max_generation=500, need_play=False)
        g, s = o.get_actual_generation()
        if s:
            generations.append(g)
        else:
            s_count += 1
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("procession: " + str(i + 1))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    print("generations:")
    print(generations)

    max_value = max(generations)
    counts = [0 for i in range(max_value)]
    for generation in generations:
        counts[generation] += 1

    print("counts:")
    print(counts)
    print("obtain failed:")
    print(s_count)
