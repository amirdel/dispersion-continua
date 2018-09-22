import numpy as np
import bisect as bs

def get_time_dx_array_with_frequency(dt_input, v_input, delta_t):
    """
    function to find the dx array for the in time with dt increments of deltaT
    """
    freq_list = []
    dx_list = []
    t_array = np.hstack(([0.0], np.cumsum(dt_input)))
    x_array = np.hstack(([0.0], np.cumsum(np.multiply(dt_input, v_input))))
    t_target = delta_t
    x_start = 0.0
    while t_target <= t_array[-1]:
        #print "t_target: ", t_target
        idx_t = bs.bisect_left(t_array, t_target)
        #print "idx_t:", idx_t
        closest_smaller_time = t_array[idx_t -1]
        dx = x_array[idx_t-1] - x_start
        x_correction = (t_target - closest_smaller_time)*v_input[idx_t - 1]
        dx += x_correction
        dx_list.append(dx)
        freq_list.append(1)
        x_start += dx
        t_target += delta_t
        #take care of repetition
        closest_larger_time = t_array[idx_t]
        n_repeat = np.floor((closest_larger_time - t_target)/delta_t)
        if n_repeat > 0:
            repeating_dx = v_input[idx_t - 1]*delta_t
            dx_list.append(repeating_dx)
            freq_list.append(n_repeat)
            x_start += n_repeat*repeating_dx
            t_target += n_repeat*delta_t
    return dx_list, freq_list


def average_dx_dy_array_with_freq_v(dt_input, vx_input, vy_input, x_start, y_start, delta_t):
    """
    function to find the dx, dy, number of repeats array for the in time with dt increments of deltaT.
    The instantaneous velocity at the end of each averaging window will also be saved.
    output theta will be in radians form -pi to pi
    :param dt_input:
    :param vx_input:
    :param vy_input:
    :param x_start:
    :param y_start:
    :param delta_t:
    :return:
    """
    freq_list, dx_list, dy_list, v_list = [[] for _ in range(4)]
    v_input = np.sqrt(vx_input**2 + vy_input**2)
    t_array = np.hstack(([0.0], np.cumsum(dt_input)))
    #x_array , y_array include the x,y location of the particle
    x_array = x_start + np.hstack((0.0, np.cumsum(np.multiply(dt_input, vx_input))))
    y_array = y_start + np.hstack((0.0, np.cumsum(np.multiply(dt_input, vy_input))))
    t_target = delta_t
    while t_target <= t_array[-1]:
        # bisect left result is >=1, since time starts from zero
        idx_t = bs.bisect_left(t_array, t_target)
        closest_smaller_time = t_array[idx_t -1]
        #advance x
        dx = x_array[idx_t-1] - x_start
        x_correction = (t_target - closest_smaller_time)*vx_input[idx_t - 1]
        dx += x_correction
        dx_list.append(dx)
        x_start += dx
        #advance y
        dy = y_array[idx_t - 1] - y_start
        y_correction = (t_target - closest_smaller_time) * vy_input[idx_t - 1]
        dy += y_correction
        dy_list.append(dy)
        y_start += dy
        #add one to frequency
        freq_list.append(1)
        # save the velocity
        v_list.append(v_input[idx_t-1])
        t_target += delta_t
        #take care of repetition
        closest_larger_time = t_array[idx_t]
        n_repeat = np.floor((closest_larger_time - t_target)/delta_t)
        if n_repeat > 0:
            repeating_dx = vx_input[idx_t - 1]*delta_t
            repeating_dy = vy_input[idx_t - 1]*delta_t
            dx_list.append(repeating_dx)
            dy_list.append(repeating_dy)
            freq_list.append(n_repeat)
            v_list.append(n_repeat)
            x_start += n_repeat*repeating_dx
            y_start += n_repeat*repeating_dy
            t_target += n_repeat*delta_t
    return dx_list, dy_list, v_list, freq_list


def get_time_dx_dy_array_with_freq(dt_input, vx_input, vy_input, x_start, y_start, delta_t):
    """
    function to find the dx, dy, number of repeats array for the in time with dt increments of deltaT.
    output theta will be in radians form -pi to pi
    :param dt_input:
    :param vx_input:
    :param vy_input:
    :param x_start:
    :param y_start:
    :param delta_t:
    :return:
    """
    freq_list, dx_list, dy_list = [[] for _ in range(3)]
    t_array = np.hstack(([0.0], np.cumsum(dt_input)))
    #x_array , y_array include the x,y location of the particle
    x_array = x_start + np.hstack((0.0, np.cumsum(np.multiply(dt_input, vx_input))))
    y_array = y_start + np.hstack((0.0, np.cumsum(np.multiply(dt_input, vy_input))))
    t_target = delta_t
    while t_target <= t_array[-1]:
        idx_t = bs.bisect_left(t_array, t_target)
        closest_smaller_time = t_array[idx_t -1]
        #advance x
        dx = x_array[idx_t-1] - x_start
        x_correction = (t_target - closest_smaller_time)*vx_input[idx_t - 1]
        dx += x_correction
        dx_list.append(dx)
        x_start += dx
        #advance y
        dy = y_array[idx_t - 1] - y_start
        y_correction = (t_target - closest_smaller_time) * vy_input[idx_t - 1]
        dy += y_correction
        dy_list.append(dy)
        y_start += dy
        #add one to frequency
        freq_list.append(1)
        t_target += delta_t
        #take care of repetition
        closest_larger_time = t_array[idx_t]
        n_repeat = np.floor((closest_larger_time - t_target)/delta_t)
        if n_repeat > 0:
            repeating_dx = vx_input[idx_t - 1]*delta_t
            repeating_dy = vy_input[idx_t - 1]*delta_t
            dx_list.append(repeating_dx)
            dy_list.append(repeating_dy)
            freq_list.append(n_repeat)
            x_start += n_repeat*repeating_dx
            y_start += n_repeat*repeating_dy
            t_target += n_repeat*delta_t
    return dx_list, dy_list, freq_list


def get_time_dx_dy_array_with_freq_nocumsum(t_array, x_array, y_array, vx_input, vy_input,
                                            x_start, y_start, delta_t):
    """
    function to find the dx, dy, number of repeats array for the in time with dt increments of deltaT
    output theta will be in radians form -pi to pi
    :param dt_input:
    :param vx_input:
    :param vy_input:
    :param x_start:
    :param y_start:
    :param delta_t:
    :return:
    """
    freq_list = []
    dx_list = []
    dy_list = []
    t_target = delta_t
    while t_target <= t_array[-1]:
        idx_t = bs.bisect_left(t_array, t_target)
        closest_smaller_time = t_array[idx_t -1]
        #advance x
        dx = x_array[idx_t-1] - x_start
        x_correction = (t_target - closest_smaller_time)*vx_input[idx_t - 1]
        dx += x_correction
        dx_list.append(dx)
        x_start += dx
        #advance y
        dy = y_array[idx_t - 1] - y_start
        y_correction = (t_target - closest_smaller_time) * vy_input[idx_t - 1]
        dy += y_correction
        dy_list.append(dy)
        y_start += dy
        #add one to frequency
        freq_list.append(1)
        t_target += delta_t
        #take care of repetition
        closest_larger_time = t_array[idx_t]
        n_repeat = np.floor((closest_larger_time - t_target)/delta_t)
        if n_repeat > 0:
            repeating_dx = vx_input[idx_t - 1]*delta_t
            repeating_dy = vy_input[idx_t - 1]*delta_t
            dx_list.append(repeating_dx)
            dy_list.append(repeating_dy)
            freq_list.append(n_repeat)
            x_start += n_repeat*repeating_dx
            y_start += n_repeat*repeating_dy
            t_target += n_repeat*delta_t
    return dx_list, dy_list, freq_list

def remove_duplicate(dx_input, freq_input):
    """
    function to remove duplicate values from frequency array after conversion.
    e.g. dx = [1, 1, 1, 3] --> dx_out = [1, 3]
          f = [1, 8, 1, 4]     f_out  = [10, 4]
    """
    dx_out = []
    freq_out = []
    i = 1
    val = dx_input[0]
    freq = freq_input[0]

    while i < len(dx_input):
        new_val = dx_input[i]
        #print 'new val: ', new_val, val
        if new_val == val:
            freq += freq_input[i]
        else:
            dx_out.append(val)
            freq_out.append(freq)
            val = new_val
            freq = freq_input[i]
        i += 1
    dx_out.append(new_val)
    freq_out.append(freq)
    return np.array(dx_out), np.array(freq_out, dtype=np.int)


def remove_duplicate_xy(dx_input, dy_input, freq_input):
    """
    function to remove duplicate values from frequency array after conversion.
    e.g. dx = [1, 1, 1, 3] --> dx_out = [1, 3]
          f = [1, 8, 1, 4]     f_out  = [10, 4]
    """
    dx_out, dy_out, freq_out = [[] for _ in range(3)]
    i = 1
    x_val = dx_input[0]
    y_val = dy_input[0]
    freq = freq_input[0]
    while i < len(dx_input):
        new_x_val = dx_input[i]
        new_y_val = dy_input[i]
        #print 'new val: ', new_val, val
        if new_x_val == x_val:
            freq += freq_input[i]
        else:
            dx_out.append(x_val)
            dy_out.append(y_val)
            freq_out.append(freq)
            x_val = new_x_val
            y_val = new_y_val
            freq = freq_input[i]
        i += 1
    dx_out.append(x_val)
    dy_out.append(y_val)
    freq_out.append(freq)
    assert(len(dx_out) == len(dy_out))
    assert(len(dx_out) == len(freq_out))
    return np.array(dx_out), np.array(dy_out), np.array(freq_out, dtype=np.int)


def remove_duplicate_xyv(dx_input, dy_input, v_input, freq_input):
    """
    function to remove duplicate values from frequency array after conversion.
    e.g. dx = [1, 1, 1, 3] --> dx_out = [1, 3]
          f = [1, 8, 1, 4]     f_out  = [10, 4]
    """
    dx_out, dy_out, freq_out, v_out = [[] for _ in range(4)]
    i = 1
    x_val = dx_input[0]
    y_val = dy_input[0]
    freq = freq_input[0]
    v_val = v_input[0]
    while i < len(dx_input):
        new_x_val = dx_input[i]
        new_y_val = dy_input[i]
        new_v_val = v_input[i]
        # check if this is a repeating value
        if new_x_val == x_val and new_y_val == y_val:
            freq += freq_input[i]
        else:
            dx_out.append(x_val)
            dy_out.append(y_val)
            v_out.append(v_val)
            freq_out.append(freq)
            x_val = new_x_val
            y_val = new_y_val
            v_val = new_v_val
            freq = freq_input[i]
        i += 1
    dx_out.append(x_val)
    dy_out.append(y_val)
    v_out.append(v_val)
    freq_out.append(freq)
    assert(len(dx_out) == len(dy_out))
    assert(len(dx_out) == len(freq_out))
    assert (len(dx_out) == len(v_out))
    return np.array(dx_out), np.array(dy_out), np.array(v_out), np.array(freq_out, dtype=np.int)