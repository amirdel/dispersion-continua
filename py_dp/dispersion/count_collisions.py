def count_collisions(time_array, t_end):
    """
    find the number of collisions before a specified end time, t_end
    :param time_array: (n_particles, n_steps) array each row contains times of collision for one particle
    :param t_end: end time
    :return:
    """
    return len(time_array[time_array <= t_end])

