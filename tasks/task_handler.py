from evolution.bean.attacker import *
from tasks.task_inform import *


def run_imply():
    task = Logic(method_type=MethodType.N, logic_type=LogicType.IMPLY, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "IMPLY", MethodType.N)

    task = Logic(method_type=MethodType.FS, logic_type=LogicType.IMPLY, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "IMPLY", MethodType.FS)

    task = Logic(method_type=MethodType.BI, logic_type=LogicType.IMPLY, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "IMPLY", MethodType.BI)

    task = Logic(method_type=MethodType.GS, logic_type=LogicType.IMPLY, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "IMPLY", MethodType.GS)


def run_nand():
    task = Logic(method_type=MethodType.N, logic_type=LogicType.NAND, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NAND", MethodType.N)

    task = Logic(method_type=MethodType.FS, logic_type=LogicType.NAND, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NAND", MethodType.FS)

    task = Logic(method_type=MethodType.BI, logic_type=LogicType.NAND, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NAND", MethodType.BI)

    task = Logic(method_type=MethodType.GS, logic_type=LogicType.NAND, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NAND", MethodType.GS)


def run_nor():
    task = Logic(method_type=MethodType.N, logic_type=LogicType.NOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NOR", MethodType.N)

    task = Logic(method_type=MethodType.FS, logic_type=LogicType.NOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NOR", MethodType.FS)

    task = Logic(method_type=MethodType.BI, logic_type=LogicType.NOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NOR", MethodType.BI)

    task = Logic(method_type=MethodType.GS, logic_type=LogicType.NOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "NOR", MethodType.GS)


def run_xor():
    task = Logic(method_type=MethodType.N, logic_type=LogicType.XOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "XOR", MethodType.N)

    task = Logic(method_type=MethodType.FS, logic_type=LogicType.XOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "XOR", MethodType.FS)

    task = Logic(method_type=MethodType.BI, logic_type=LogicType.XOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "XOR", MethodType.BI)

    task = Logic(method_type=MethodType.GS, logic_type=LogicType.XOR, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "XOR", MethodType.GS)


def run_cart_pole_v0():
    task = Game(method_type=MethodType.N, game_type=GameType.CartPole_v0, episode_steps=300, episode_generation=10,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", MethodType.N)

    task = Game(method_type=MethodType.FS, game_type=GameType.CartPole_v0, episode_steps=300, episode_generation=10,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", MethodType.FS)

    task = Game(method_type=MethodType.BI, game_type=GameType.CartPole_v0, episode_steps=300, episode_generation=10,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", MethodType.BI)

    task = Game(method_type=MethodType.GS, game_type=GameType.CartPole_v0, episode_steps=300, episode_generation=10,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", MethodType.GS)


def run_cart_pole_v0_with_attack():
    attacker = CartPole_v0_Attacker(attack_type=AttackType.Gaussian, gaussian_peak=1000)
    noise_level = 1

    task = Game(method_type=MethodType.FS, game_type=GameType.CartPole_v0, episode_steps=500, episode_generation=20,
                attacker=attacker, noise_level=noise_level, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", MethodType.FS)

    task = Game(method_type=MethodType.BI, game_type=GameType.CartPole_v0, episode_steps=300, episode_generation=10,
                attacker=attacker, noise_level=noise_level, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", MethodType.BI)

    task = Game(method_type=MethodType.GS, game_type=GameType.CartPole_v0, episode_steps=300, episode_generation=10,
                attacker=attacker, noise_level=noise_level, max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "CartPole_v0", MethodType.GSS)


def run_lunar_lander_v0():
    task = Game(method_type=MethodType.N, game_type=GameType.LunarLander_v2, episode_steps=100, episode_generation=2,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "LunarLander_v2", MethodType.N)

    task = Game(method_type=MethodType.FS, game_type=GameType.LunarLander_v2, episode_steps=100, episode_generation=2,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "LunarLander_v2", MethodType.FS)

    task = Game(method_type=MethodType.BI, game_type=GameType.LunarLander_v2, episode_steps=100, episode_generation=2,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "LunarLander_v2", MethodType.BI)

    task = Game(method_type=MethodType.GS, game_type=GameType.LunarLander_v2, episode_steps=100, episode_generation=2,
                max_generation=500, display_results=False)
    generations, counts = task.run(1000)
    save_distribution(counts, "../output/", "LunarLander_v2", MethodType.GS)
