from random import choice

import numpy as np
from matplotlib import pyplot as plt

def show_image(arr: np.array, title = ""):
    plt.figure(figsize = (4, 4))
    plt.imshow(arr, cmap = "gray", vmin = 0, vmax = 1)
    plt.axis(False)
    
    if len(title) > 0:
        plt.title(title, fontsize = 14)

    plt.show()

def get_pred_arr(filter_pos: np.array, fitler_arr: np.array, output_size: tuple):
    output_arr = np.zeros(shape = output_size, dtype = np.float64)
    filter_size: tuple = fitler_arr.shape

    for pos in filter_pos:
        filter_slice = slice(pos[0], pos[0] + filter_size[0]), slice(pos[1], pos[1] + filter_size[1])
        output_arr[filter_slice] += fitler_arr
    
    return output_arr.clip(max = 1)[filter_size[0] - 1: - filter_size[0] + 1, filter_size[1] - 1: - filter_size[1] + 1]

def get_error(true_arr, pred_arr):
    return np.abs(true_arr - pred_arr).sum().round(3)

def cross_filters(filter_pos1: np.array, filter_pos2: np.array):
    filter_count = len(filter_pos1)
    return np.stack(
        (filter_pos1, filter_pos2)
    )[
        np.random.randint(0, 2, size = filter_count),
        range(filter_count)
    ]

def show_best_worst(
    pop_filter_pos: np.array,
    img_arr: np.array,
    fitler_arr: np.array,
    output_size: tuple,
):
    output_arrs = [
        get_pred_arr(filter_pos, fitler_arr, output_size)
        for filter_pos in pop_filter_pos
    ]

    errors = [get_error(img_arr, output_arr) for output_arr in output_arrs]
    print(f"Best score: {min(errors)} | Worst score: {max(errors)}")
    filter_ranks = np.argsort(errors)

    show_image(output_arrs[filter_ranks[0]], title = f"Error: {errors[filter_ranks[0]]}")
    show_image(output_arrs[filter_ranks[-1]], title = f"Error: {errors[filter_ranks[-1]]}")

def iterate_gen(
    pop_filter_pos: np.array,
    img_arr: np.array,
    fitler_arr: np.array,
    output_size: tuple,
    elitism: int,
    mutate_prob: float,
    filter_max_pos: tuple[int]
):
    pop_size = len(pop_filter_pos)
    filter_count = len(pop_filter_pos[0])

    output_arrs = [
        get_pred_arr(filter_pos, fitler_arr, output_size)
        for filter_pos in pop_filter_pos
    ]

    errors = [get_error(img_arr, output_arr) for output_arr in output_arrs]
    print(f"Best score: {min(errors)} | Worst score: {max(errors)}")
    filter_ranks = np.argsort(errors)

    new_pop_filter_pos = []

    for i in filter_ranks[:elitism]:
        new_pop_filter_pos.append(pop_filter_pos[i])

    for _ in range(pop_size - elitism):
        new_filter = cross_filters(choice(pop_filter_pos), choice(pop_filter_pos))

        if np.random.random() < mutate_prob:
            new_filter[np.random.randint(filter_count)] = [
                np.random.randint(filter_max_pos[0]), 
                np.random.randint(filter_max_pos[1])
            ]
            
        new_pop_filter_pos.append(new_filter)
    
    return new_pop_filter_pos