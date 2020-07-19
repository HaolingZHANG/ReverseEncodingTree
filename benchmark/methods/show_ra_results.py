import csv
import numpy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_rastrigin(path):
    matrix = []
    with open(path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            matrix.append(list(map(float, row)))

    x_axis = numpy.arange(0, 200, 1)
    y_axis = numpy.arange(0, 200, 1)
    x_axis, y_axis = numpy.meshgrid(x_axis, y_axis)

    return x_axis, y_axis, numpy.array(matrix)


def show_rastrigin(path, save_path=None):
    x_axis, y_axis, z_axis = get_rastrigin(path)
    max_height = -10
    min_height = 60
    min_position = [0, 0]
    max_position = [0, 0]
    for row in range(len(z_axis)):
        for col in range(len(z_axis[row])):
            if z_axis[row][col] < min_height:
                min_height = z_axis[row][col]
                min_position = [row, col]
            if z_axis[row][col] > max_height:
                max_height = z_axis[row][col]
                max_position = [row, col]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.plot_surface(x_axis, y_axis, z_axis, rstride=1, cstride=1, cmap='rainbow', alpha=0.8)
    ax.scatter([min_position[1]], [min_position[0]], [min_height], c='indigo')
    ax.scatter([max_position[1]], [max_position[0]], [max_height], c='red')
    ax.contourf(x_axis, y_axis, z_axis,
                zdir='z', offset=-100, cmap='rainbow')

    ax.view_init(35, -150)
    ax.set_zlim(-100, 60)

    if save_path is not None:
        plt.savefig(save_path + "ra_clime.png", format='png', bbox_inches='tight', transparent=True, dpi=600)

    plt.show()


def clime_by_generation(population_recorders,
                        title, generation, is_final=False, mount_path="mount_everest.csv", save_path=None):
    x_axis, y_axis, z_axis = get_rastrigin(mount_path)

    fig = plt.figure()
    ax = Axes3D(fig)

    if generation == 0:
        population_recorder = population_recorders[generation]
        ax.set_title(title + ": (initialized" +
                     " generations to reach " + str(numpy.max(population_recorder[2])) + ")")
    elif is_final:
        population_recorder = population_recorders[-1]
        ax.set_title(title + ": (" + str(generation + 1) +
                     " generations to reach " + str(numpy.max(population_recorder[2])) + " finally)")
    else:
        population_recorder = population_recorders[generation]
        ax.set_title(title + ": (" + str(generation + 1) +
                     " generations to reach " + str(numpy.max(population_recorder[2])) + ")")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.plot_surface(x_axis, y_axis, z_axis, rstride=1, cstride=1, cmap='rainbow', alpha=0.3)
    ax.scatter(population_recorder[1], population_recorder[0], population_recorder[2], c='black')
    ax.contourf(x_axis, y_axis, z_axis, zdir='z', offset=-100, cmap='rainbow', alpha=0.3)
    ax.scatter(population_recorder[1], population_recorder[0],
               [-100 for _ in range(len(population_recorder[0]))], c='black')

    ax.view_init(35, -150)
    ax.set_zlim(-100, 60)

    if save_path is not None:
        plt.savefig(save_path + title + "." + str(generation + 1) + ".png", format='png', bbox_inches='tight',
                    transparent=True, dpi=600)
    plt.show()
