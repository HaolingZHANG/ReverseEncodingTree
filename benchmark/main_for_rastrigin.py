from benchmark.methods import inherent
from benchmark.methods.evolutor import GA, PBIL, BI, TRI
from benchmark.methods.show_me_results import clime_by_generation, get_mount_everest


def obtain_ga(start_position, stride):
    _, _, landscape = get_mount_everest("./dataset/rastrigin.csv")
    function = GA(size=20, scope=len(landscape), start_position=start_position, stride=stride)
    function.evolute(terrain=landscape, generations=6, evolute_type=inherent.MAX)

    for i in range(5):
        clime_by_generation(function.recorder.get_result(), "GA", i, False,
                            mount_path="./dataset/rastrigin.csv", save_path="./results/")


def obtain_pbil():
    _, _, landscape = get_mount_everest("./dataset/rastrigin.csv")
    function = PBIL(size=20, scope=len(landscape))
    function.evolute(terrain=landscape, generations=6, evolute_type=inherent.MAX)

    for i in range(5):
        clime_by_generation(function.recorder.get_result(), "PBIL", i, False,
                            mount_path="./dataset/rastrigin.csv", save_path="./results/")


def obtain_bi():
    _, _, landscape = get_mount_everest("./dataset/rastrigin.csv")
    function = BI(size=20, init_interval=40, min_interval=3, scope=len(landscape))
    function.evolute(terrain=landscape, generations=6, evolute_type=inherent.MAX)

    for i in range(5):
        clime_by_generation(function.recorder.get_result(), "BI", i, False,
                            mount_path="./dataset/rastrigin.csv", save_path="./results/")


def obtain_tri():
    _, _, landscape = get_mount_everest("./dataset/rastrigin.csv")
    function = TRI(size=20, init_interval=40, min_interval=3, scope=len(landscape))
    function.evolute(terrain=landscape, generations=6, evolute_type=inherent.MAX)

    for i in range(5):
        clime_by_generation(function.recorder.get_result(), "TRI", i, False,
                            mount_path="./dataset/rastrigin.csv", save_path="./results/")


def get_min(landscape):
    height = landscape[0][0]
    position = [0, 0]
    for row in range(len(landscape)):
        for col in range(len(landscape[row])):
            if landscape[row][col] < height:
                height = landscape[row][col]
                position = [row, col]

    return position, height


if __name__ == '__main__':
    _, _, fitness_landscape = get_mount_everest("./dataset/rastrigin.csv")
    min_position, min_height = get_min(fitness_landscape)
    obtain_ga(min_position, 15)
    obtain_pbil()
    obtain_bi()
    obtain_tri()
