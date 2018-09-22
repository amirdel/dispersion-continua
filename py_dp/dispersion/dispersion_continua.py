# Copyright 2017 Amir Hossein Delgoshaie, amirdel@stanford.edu
#
# Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee
# is hereby granted, provided that the above copyright notice and this permission notice appear in all
# copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE
# INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE
# FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
# ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import numpy as np
import math as math
import warnings

class dispersionSystemContinua():
    def __init__(self, grid, n_particles, n_steps, tracking_type='dt'):
        self.grid = grid
        self.p = grid.pressure
        self.n_particles = n_particles
        self.n_steps = n_steps
        self.init_cells = np.zeros(n_particles)
        self.time_array = np.zeros((n_particles, n_steps + 1))
        self.cell_nr_array = np.zeros((n_particles, n_steps + 1), dtype=np.int)
        self.x_array = np.zeros((n_particles, n_steps + 1))
        self.y_array = np.zeros((n_particles, n_steps + 1))
        self.time_array = np.zeros((n_particles,n_steps+1))
        # self.init_particles(injection_type)
        # there are two possibilities for tracking the particles
        # 1. track the particle as it exits each cell -> tracking_type = 'exit'
        # 2. track the particle given a delta t -> tracking_type = 'dt'
        self.tracking_type = tracking_type

    def init_particles_one_middle(self):
        grid = self.grid
        x_array = grid.pores.x
        y_array = grid.pores.y
        slab_pores = np.where(x_array == np.amin(x_array))[0]
        y_slab = y_array[slab_pores]
        ymid = np.amax(y_slab) / 2
        dist = abs(y_slab - ymid)
        idxChosen = np.argmin(dist)
        init_pores = np.array(idxChosen * np.ones(self.n_particles, dtype=int))
        self.x_array[:, 0] = x_array[init_pores]
        self.y_array[:, 0] = y_array[init_pores]
        self.cell_nr_array[:, 0] = init_pores

    def init_particles_left_boundary(self):
        grid = self.grid
        x_array = grid.pores.x
        y_array = grid.pores.y
        n_particles = self.n_particles
        init_pores = range(n_particles)
        self.x_array[:, 0] = x_array[init_pores]
        self.y_array[:, 0] = y_array[init_pores]
        self.cell_nr_array[:, 0] = init_pores

    def init_particles_left_buffered(self, n_buffer):
        grid = self.grid
        m = grid.m
        y_array = grid.pores.y
        x_array = grid.pores.x
        n_particles = self.n_particles
        dx = grid.dx
        left_boundary = range(m)
        y_left = y_array[left_boundary]
        y_max = grid.ly
        buffer_length = n_buffer*dx
        init_cells = [i for i in range(m) if (y_left[i]>buffer_length and  y_left[i] < y_max - buffer_length)]
        counter, loc = len(init_cells), 0
        init_size = counter
        while counter<n_particles:
            init_cells.append(m + init_cells[loc % init_size])
            counter += 1
            loc += 1
        self.x_array[:, 0] = x_array[init_cells]
        self.y_array[:, 0] = y_array[init_cells]
        self.cell_nr_array[:, 0] = init_cells

    def init_particles_ly_distance(self, n_buffer_ly, ly):
        grid = self.grid
        y_array = grid.pores.y
        x_array = grid.pores.x
        start_cell = n_buffer_ly*ly
        end_cell = grid.m - n_buffer_ly*ly
        init_cells = np.arange(start_cell, end_cell, ly)
        # print 'n_particles is set to: ', len(init_cells)
        # self.n_particles = len(init_cells)
        self.x_array[:, 0] = x_array[init_cells]
        self.y_array[:, 0] = y_array[init_cells]
        self.cell_nr_array[:, 0] = init_cells

    def find_exit_conditions_1d(self, dl, x, x1, f1, v1, x2, f2, v2):
        exit, exit_idx, xe, v, ve, dt = False, None, None, None, None, None
        log_dt = True
        A = (v2 - v1) / dl
        if A:
            A_inv = 1.0 / A
        v = v1 + (x - x1) * A
        if v1*v2 > 0:
            exit = True
            if v1 > 0:
                ve, exit_idx, xe = v2, f2, x2
            else:
                ve, exit_idx, xe = v1, f1, x1
            if A == 0:
                log_dt = False
                dt = abs((x-xe) / v)
        elif v1>0 and v2<0:
            exit, log_dt = False, False
        elif v1<0 and v2>0:
            exit = True
            x_stagnation = x1 - v1*A_inv
            if x > x_stagnation:
                ve, exit_idx, xe = v2, f2, x2
            elif x < x_stagnation:
                ve, exit_idx, xe = v1, f1, x1
            else:
                exit, log_dt = False, False
        elif v1 == 0 and v2 > 0:
            exit, ve, exit_idx, xe = True, v2, f2, x2
        elif v1 < 0 and v2 == 0:
            exit, ve, exit_idx, xe = True, v1, f1, x1
        elif v1 == 0 and v2 == 0:
            exit, log_dt = False, False
        if log_dt:
            dt = math.log(ve / v) * A_inv
        return exit, exit_idx, xe, v, ve, dt

    def calculate_exit_location(self, x, xl, dx, vl, vr, v, dt_e):
        if vl != vr:
            A = (vr-vl)/dx
            A_inv = 1.0/A
            xe = xl + A_inv * (v * math.exp(A * dt_e) - vl)
        else:
            xe = x + vr*dt_e
        return xe

    def find_exit_conditions(self, start_cell, xp, yp, tp, x_periodic = 0.0, y_periodic = 0.0):
        grid = self.grid
        dx, dy = grid.dx, grid.dy
        xl = x_periodic + grid.pores.x[start_cell] - dx / 2
        yb = y_periodic + grid.pores.y[start_cell] - dy / 2
        xr = x_periodic + grid.pores.x[start_cell] + dx / 2
        yt = y_periodic + grid.pores.y[start_cell] + dy / 2
        cell_faces = grid.facelist_array[start_cell]
        ngh_cells = grid.nghlist_array[start_cell]
        face_velocities = grid.face_velocities[cell_faces]
        vl, vr, vb, vt = face_velocities[0], face_velocities[1], face_velocities[2], face_velocities[3]
        exit_x, exit_idx_x, xe, v_x, ve_x, dt_x = self.find_exit_conditions_1d(dx, xp, xl, 0, vl, xr, 1, vr)
        exit_y, exit_idx_y, ye, v_y, ve_y, dt_y = self.find_exit_conditions_1d(dy, yp, yb, 2, vb, yt, 3, vt)
        calc_ye, calc_xe = False, False
        if exit_x and exit_y:
            if dt_x < dt_y:
                dt_e, exit_idx = dt_x, exit_idx_x
                calc_ye = True
            else:
                dt_e, exit_idx = dt_y, exit_idx_y
                calc_xe = True
        elif exit_x:
            dt_e, exit_idx = dt_x, exit_idx_x
            calc_ye = True
        elif exit_y:
            dt_e, exit_idx = dt_y, exit_idx_y
            calc_xe = True
        else:
            exit_idx = None
            dt_e = 0
            warnings.warn('the particle will never exit')
        if calc_xe:
            xe = self.calculate_exit_location(xp, xl, dx, vl, vr, v_x, dt_e)
        if calc_ye:
            ye = self.calculate_exit_location(yp, yb, dy, vb, vt, v_y, dt_e)
        if exit_idx is not None:
            exit_cell = ngh_cells[exit_idx]
            exit_face = cell_faces[exit_idx]
        else:
            exit_cell = None
            exit_face = None
        return exit_cell, exit_face, xe, ye, tp+dt_e

    def integrate_path(self, start_cell, xp, yp, dt):
        # TODO
        pass
        # return end_cell, xe, ye

    def follow_all_particles(self):
        if self.tracking_type == 'dt':
            self.follow_all_particles_dt()
        else:
            self.follow_all_particles_exit()

    def updatePeriod(self, start_cell, exit_face, nx_period, ny_period):
        y_faces = self.grid.y_faces
        updwn_cells = self.grid.updwn_cells
        if y_faces[exit_face]:
            if start_cell == updwn_cells[exit_face][0]:
                ny_period += 1
            else:
                ny_period -= 1
        else: # x face
            if start_cell == updwn_cells[exit_face][0]:
                nx_period += 1
            else:
                nx_period -= 1
        return nx_period, ny_period


    def follow_all_particles_exit(self):
        grid = self.grid
        periodicFaces = self.grid.periodic_faces
        x_array , y_array, t_array = self.x_array, self.y_array, self.time_array
        cell_nr_array = self.cell_nr_array
        lx, ly = grid.lx, grid.ly
        remove_list = []
        for i in range(self.n_particles):
            #start from the begining of the domain
            nx_period, ny_period = 0, 0
            xs, ys, ts, start_cell = x_array[i,0], y_array[i,0], t_array[i,0], cell_nr_array[i,0]
            for step in range(1, self.n_steps+1):
                x_periodic, y_periodic = nx_period*lx, ny_period*ly
                exit_cell, exit_face, xe, ye, te = self.find_exit_conditions(start_cell, xs, ys, ts,
                                                                             x_periodic, y_periodic)
                if exit_cell is None:
                    remove_list.append(i)
                    print 'particle did not exit!'
                    break
                # assuming boundary type is periodic
                if exit_face in periodicFaces:
                    nx_period, ny_period = self.updatePeriod(start_cell, exit_face, nx_period, ny_period)
                x_array[i,step] = xe
                y_array[i,step] = ye
                t_array[i,step] = te
                cell_nr_array[i,step] = exit_cell
                xs, ys, ts, start_cell = xe, ye, te, exit_cell
        # remove particles that were blocked
        if len(remove_list):
            self.x_array = np.delete(self.x_array, remove_list, 0)
            self.y_array = np.delete(self.y_array, remove_list, 0)
            self.time_array = np.delete(self.time_array, remove_list, 0)
            self.cell_nr_array = np.delete(self.cell_nr_array, remove_list, 0)

    def follow_all_particles_dt(self, dt):
        x_array, y_array, t_array = self.x_array, self.y_array, self.time_array
        cell_nr_array = self.cell_nr_array
        n_steps = self.n_steps
        for i in range(self.n_particles):
            xs, ys, ts, start_cell = x_array[i, 0], y_array[i, 0], t_array[i, 0], cell_nr_array[i, 0]
            for step in range(1, n_steps):
                end_cell, exit_face, xe, ye = self.integrate_path(start_cell, xs, ys, dt)
                te = ts + dt
                x_array[i, step] = xe
                y_array[i, step] = ye
                t_array[i, step] = te
                cell_nr_array[i, step] = end_cell
                xs, ys, ts, start_cell = xe, ye, te, end_cell

def find_compatible_n_particles(grid_size, ly, n_buffer_ly):
    """
    find the number of particles, n, that would be consistent with having n trajectories
    distances by ly and buffered by n_buffer_ly*ly from top and bottom boundaries
    :param grid:
    :param ly: correlation length in number of grid cells
    :param n_buffer_ly: number of ly to buffer from top and bottom boundary (buffer = n_buffer_ly*ly cells)
    :return:
    """
    start_cell = n_buffer_ly * ly
    end_cell = grid_size - n_buffer_ly * ly
    init_cells = np.arange(start_cell, end_cell, ly)
    return len(init_cells)