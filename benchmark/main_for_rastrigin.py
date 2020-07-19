from ReverseEncodingTree.benchmark.methods import inherent
from ReverseEncodingTree.benchmark.methods.evolutor import GA, PBIL, BI, CMAES
from ReverseEncodingTree.benchmark.methods.show_ra_results import clime_by_generation, get_rastrigin


def obtain_ga(start_position, stride):
    _, _, landscape = get_rastrigin("./dataset/rastrigin.csv")
    function = GA(size=90, scope=len(landscape), start_position=start_position, stride=stride)
    function.evolute(terrain=landscape, generations=200, evolute_type=inherent.MAX)

    for i in [0, 49, 99, 149, 199]:
        clime_by_generation(function.recorder.get_result(), "GA", i, False,
                            mount_path="./dataset/rastrigin.csv", save_path="./results/")


def obtain_pbil():
    _, _, landscape = get_rastrigin("./dataset/rastrigin.csv")
    function = PBIL(size=90, scope=len(landscape))
    function.evolute(terrain=landscape, generations=80, evolute_type=inherent.MAX)

    for i in [0, 19, 39, 59, 79]:
        clime_by_generation(function.recorder.get_result(), "PBIL", i, False,
                            mount_path="./dataset/rastrigin.csv", save_path="./results/")


def obtain_cmaes(start_position, stride):
    _, _, landscape = get_rastrigin("./dataset/rastrigin.csv")
    function = CMAES(size=90, scope=len(landscape), start_position=start_position, stride=stride)
    function.evolute(terrain=landscape, generations=20, evolute_type=inherent.MAX)

    for i in [0, 4, 9, 14, 19]:
        clime_by_generation(function.recorder.get_result(), "CMA-ES", i, False,
                            mount_path="./dataset/rastrigin.csv", save_path="./results/")


def obtain_bi():
    _, _, landscape = get_rastrigin("./dataset/rastrigin.csv")
    function = BI(size=10, init_interval=40, min_interval=3, scope=len(landscape))
    function.evolute(terrain=landscape, generations=8, evolute_type=inherent.MAX)

    for i in [0, 1, 3, 5, 7]:
        clime_by_generation(function.recorder.get_result(), "RET", i, False,
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
    _, _, fitness_landscape = get_rastrigin("./dataset/rastrigin.csv")
    min_position, min_height = get_min(fitness_landscape)
    obtain_ga(min_position, 15)
    obtain_pbil()
    obtain_cmaes(min_position, 28)
    obtain_bi()
