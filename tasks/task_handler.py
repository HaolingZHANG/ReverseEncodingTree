from evolution.bean.attacker import *
from tasks.task_inform import *


def run_imply():
    task = Logic(method_type=METHOD_TYPE.N, logic_type=LOGIC_TYPE.IMPLY, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "IMPLY", METHOD_TYPE.N)

    task = Logic(method_type=METHOD_TYPE.FS, logic_type=LOGIC_TYPE.IMPLY, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "IMPLY", METHOD_TYPE.FS)

    task = Logic(method_type=METHOD_TYPE.BI, logic_type=LOGIC_TYPE.IMPLY, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "IMPLY", METHOD_TYPE.BI)

    task = Logic(method_type=METHOD_TYPE.GS, logic_type=LOGIC_TYPE.IMPLY, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "IMPLY", METHOD_TYPE.GS)


def run_nand():
    task = Logic(method_type=METHOD_TYPE.N, logic_type=LOGIC_TYPE.NAND, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NAND", METHOD_TYPE.N)

    task = Logic(method_type=METHOD_TYPE.FS, logic_type=LOGIC_TYPE.NAND, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NAND", METHOD_TYPE.FS)

    task = Logic(method_type=METHOD_TYPE.BI, logic_type=LOGIC_TYPE.NAND, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NAND", METHOD_TYPE.BI)

    task = Logic(method_type=METHOD_TYPE.GS, logic_type=LOGIC_TYPE.NAND, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NAND", METHOD_TYPE.GS)


def run_nor():
    task = Logic(method_type=METHOD_TYPE.N, logic_type=LOGIC_TYPE.NOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NOR", METHOD_TYPE.N)

    task = Logic(method_type=METHOD_TYPE.FS, logic_type=LOGIC_TYPE.NOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NOR", METHOD_TYPE.FS)

    task = Logic(method_type=METHOD_TYPE.BI, logic_type=LOGIC_TYPE.NOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NOR", METHOD_TYPE.BI)

    task = Logic(method_type=METHOD_TYPE.GS, logic_type=LOGIC_TYPE.NOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NOR", METHOD_TYPE.GS)


def run_xor():
    task = Logic(method_type=METHOD_TYPE.N, logic_type=LOGIC_TYPE.XOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "XOR", METHOD_TYPE.N)

    task = Logic(method_type=METHOD_TYPE.FS, logic_type=LOGIC_TYPE.XOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "XOR", METHOD_TYPE.FS)

    task = Logic(method_type=METHOD_TYPE.BI, logic_type=LOGIC_TYPE.XOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "XOR", METHOD_TYPE.BI)

    task = Logic(method_type=METHOD_TYPE.GS, logic_type=LOGIC_TYPE.XOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "XOR", METHOD_TYPE.GS)


def run_cart_pole_v0():
    task = Game(method_type=METHOD_TYPE.N, game_type=GAME_TYPE.CartPole_v0, episode_steps=300, episode_generation=10,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", METHOD_TYPE.N)

    task = Game(method_type=METHOD_TYPE.FS, game_type=GAME_TYPE.CartPole_v0, episode_steps=300, episode_generation=10,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", METHOD_TYPE.FS)

    task = Game(method_type=METHOD_TYPE.BI, game_type=GAME_TYPE.CartPole_v0, episode_steps=300, episode_generation=10,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", METHOD_TYPE.BI)

    task = Game(method_type=METHOD_TYPE.GS, game_type=GAME_TYPE.CartPole_v0, episode_steps=300, episode_generation=10,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", METHOD_TYPE.GS)


def run_cart_pole_v0_with_attack():
    attacker = CartPole_v0_Attacker(attack_type=ATTACK_TYPE.GaussianAvg, gaussian_peak=1000)
    noise_level = 1

    task = Game(method_type=METHOD_TYPE.FS, game_type=GAME_TYPE.CartPole_v0, episode_steps=500, episode_generation=20,
                attacker=attacker, noise_level=noise_level, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", METHOD_TYPE.FS)

    task = Game(method_type=METHOD_TYPE.BI, game_type=GAME_TYPE.CartPole_v0, episode_steps=300, episode_generation=10,
                attacker=attacker, noise_level=noise_level, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", METHOD_TYPE.BI)

    task = Game(method_type=METHOD_TYPE.GS, game_type=GAME_TYPE.CartPole_v0, episode_steps=300, episode_generation=10,
                attacker=attacker, noise_level=noise_level, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", METHOD_TYPE.GSS)


def run_lunar_lander_v0():
    task = Game(method_type=METHOD_TYPE.N, game_type=GAME_TYPE.LunarLander_v2, episode_steps=100, episode_generation=2,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "LunarLander_v2", METHOD_TYPE.N)

    task = Game(method_type=METHOD_TYPE.FS, game_type=GAME_TYPE.LunarLander_v2, episode_steps=100, episode_generation=2,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "LunarLander_v2", METHOD_TYPE.FS)

    task = Game(method_type=METHOD_TYPE.BI, game_type=GAME_TYPE.LunarLander_v2, episode_steps=100, episode_generation=2,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "LunarLander_v2", METHOD_TYPE.BI)

    task = Game(method_type=METHOD_TYPE.GS, game_type=GAME_TYPE.LunarLander_v2, episode_steps=100, episode_generation=2,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "LunarLander_v2", METHOD_TYPE.GS)