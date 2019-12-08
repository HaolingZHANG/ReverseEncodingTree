from evolution_process.bean.attacker import *
from tasks.task_inform import *


def run_nand():
    task = Logic(method_type=METHOD_TYPE.FS, logic_type=LOGIC_TYPE.NAND, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NAND", METHOD_TYPE.FS)

    task = Logic(method_type=METHOD_TYPE.BI, logic_type=LOGIC_TYPE.NAND, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NAND", METHOD_TYPE.BI)

    task = Logic(method_type=METHOD_TYPE.GSS, logic_type=LOGIC_TYPE.NAND, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NAND", METHOD_TYPE.GSS)


def run_nor():
    task = Logic(method_type=METHOD_TYPE.FS, logic_type=LOGIC_TYPE.NOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NOR", METHOD_TYPE.FS)

    task = Logic(method_type=METHOD_TYPE.BI, logic_type=LOGIC_TYPE.NOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NOR", METHOD_TYPE.BI)

    task = Logic(method_type=METHOD_TYPE.GSS, logic_type=LOGIC_TYPE.NOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NOR", METHOD_TYPE.GSS)


def run_imply():
    task = Logic(method_type=METHOD_TYPE.FS, logic_type=LOGIC_TYPE.IMPLY, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "IMPLY", METHOD_TYPE.FS)

    task = Logic(method_type=METHOD_TYPE.BI, logic_type=LOGIC_TYPE.IMPLY, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "IMPLY", METHOD_TYPE.BI)

    task = Logic(method_type=METHOD_TYPE.GSS, logic_type=LOGIC_TYPE.IMPLY, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "IMPLY", METHOD_TYPE.GSS)


def run_xor():
    task = Logic(method_type=METHOD_TYPE.FS, logic_type=LOGIC_TYPE.XOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "XOR", METHOD_TYPE.FS)

    task = Logic(method_type=METHOD_TYPE.BI, logic_type=LOGIC_TYPE.XOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "XOR", METHOD_TYPE.BI)

    task = Logic(method_type=METHOD_TYPE.GSS, logic_type=LOGIC_TYPE.XOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "XOR", METHOD_TYPE.GSS)


def run_cart_pole_v0():
    task = Game(method_type=METHOD_TYPE.FS, game_type=GAME_TYPE.CartPole_v0, episode_steps=300, episode_generation=10,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", METHOD_TYPE.FS)

    task = Game(method_type=METHOD_TYPE.BI, game_type=GAME_TYPE.CartPole_v0, episode_steps=300, episode_generation=10,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", METHOD_TYPE.BI)

    task = Game(method_type=METHOD_TYPE.GSS, game_type=GAME_TYPE.CartPole_v0, episode_steps=300, episode_generation=10,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", METHOD_TYPE.GSS)


def run_cart_pole_v0_with_attack():
    attacker = CartPole_v0_Attacker(attack_type=ATTACK_TYPE.Normal)
    noise_level = 0.5

    task = Game(method_type=METHOD_TYPE.FS, game_type=GAME_TYPE.CartPole_v0, episode_steps=300, episode_generation=10,
                attacker=attacker, noise_level=noise_level, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", METHOD_TYPE.FS)

    task = Game(method_type=METHOD_TYPE.BI, game_type=GAME_TYPE.CartPole_v0, episode_steps=300, episode_generation=10,
                attacker=attacker, noise_level=noise_level, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", METHOD_TYPE.BI)

    task = Game(method_type=METHOD_TYPE.GSS, game_type=GAME_TYPE.CartPole_v0, episode_steps=300, episode_generation=10,
                attacker=attacker, noise_level=noise_level, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", METHOD_TYPE.GSS)


if __name__ == '__main__':
    run_xor()
