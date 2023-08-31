"""
module for simulation the motion an incompressible fluid using D2Q9 Lattice Boltzmann method
"""

import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from numba import jit
from numba import vectorize, float32, int32
from numba.typed import List


class Lattice:
    """
    class creates lattice with both horizontal and vertical increment as 1

    Attributes
    -------------
    nx : int , number of column in the lattice
    ny : int , number of row in the lattice
    viscosity : float, viscosity of the fluid

    Methods
    -------------
    set_obstacle(bool_matrix) : takes a boolean matrix of the same size as lattice to represent an obstacle
    set_tao(tao) : sets the relaxation time externally

    """

    def __init__(self, nx, ny, viscosity=1 / 6.0, temperature=298):
        self.nx = nx  # number of cells, horizontal
        self.ny = ny  # number of cells, vertical

        # initialising the microscopic velocity, each coordinate holds an array representing vector[vertical,horizontal]
        self.bulk_velocity = np.zeros((nx + 2, ny + 2, 2), float)

        # self.bulk_velocity[self.ny, :] = [1, 0]

        # defining the velocity for all 9 vectors in a 1x1 cell
        #    6   2    5
        #      \ |  /
        #  3 <-  0  -> 1
        #      / | \
        #    7   4   8
        self.e = np.array(
            [
                np.array([0.0, 0.0]),
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
                np.array([-1.0, 0.0]),
                np.array([0.0, -1.0]),
                np.array([1.0, 1.0]),
                np.array([-1.0, 1.0]),
                np.array([-1.0, -1.0]),
                np.array([1.0, -1.0]),
            ]
        )

        # defining the weight of each vector, adds up to be 1
        self.w = np.array(
            [
                4 / 9.0,
                1 / 9.0,
                1 / 9.0,
                1 / 9.0,
                1 / 9.0,
                1 / 36.0,
                1 / 36.0,
                1 / 36.0,
                1 / 36.0,
            ]
        )

        self.reflect = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

        # defining the 3D array of floats holding distribution & equilibrium function of 9 vectors for each coordinates
        self.f = np.zeros((nx + 2, ny + 2, 9), np.float_)  # distribution function
        # self.eq_f = np.zeros_like(self.f)          # equilibrium function

        self.old_f = np.zeros_like(self.f)

        # defining the density, const since the fluid incompressible
        self.rho = 1

        # defining viscosity and relaxation time
        self.viscosity = viscosity
        self.relax_time = ((6 * self.viscosity) + 1) / 2

        # defining boolean matrix for an obstacle
        self.obstacle = np.zeros((nx + 2, ny + 2), bool)
        self.obs_location_list = []

        # initialising microscopic velocity so fluid flows from left to right
        self.start_point = int(self.ny / 3)
        self.end_point = int(self.ny * (2 / 3))
        self.bulk_velocity[0, 2:-2] = [0.5, 0]

        # self.start_with_all()

        self.simulate = self.simulate_

        # initialising equilibrium function using microscopic velocity
        self.collision(init=True)
        # initialising distribution using equilibrium function
        # self.initialise_f()
        # working out new microscopic velocity using newly initialised distribution function
        self.new_velocity()

        self.file = ""
        self.export = False

        self._stop = False

    def new_velocity(self):
        self._new_velocity(
            self.nx, self.ny, self.f, self.e, self.bulk_velocity, self.rho
        )

    @staticmethod
    @jit(nopython=True)
    def _new_velocity(nx, ny, f, e, bulk_velocity, rho):
        # renews the microscopic velocity with current distribution function
        for x in range(nx + 2):
            for y in range(ny + 2):
                temp = np.array([0.0])
                for i in range(9):
                    # print(self.f[x, y, i], self.e[i])
                    temp = temp + f[x, y, i] * e[i]
                    # print(temp)
                # p = density , f = distribution function, x = position (vector), t = time
                # u = microscopic velocity (vector), e = velocity vectors in the cell (vector)
                # using p(x, t) = Sigma(f(x, t) of all 9 vectors) and
                # u(x, t) * p(x, t) = Sigma(f(x, t)* e(x, t) for all 9 vectors)
                bulk_velocity[x, y] = temp / rho

    def collision(self, init=False):
        self._collision(
            self.nx,
            self.ny,
            self.rho,
            self.relax_time,
            self.bulk_velocity,
            self.f,
            self.e,
            self.w,
            init,
        )

    @staticmethod
    @jit(nopython=True)
    def _collision(nx, ny, rho, relax_time, bulk_velocity, f, e, w, init):
        # renews the equilibrium function using current microscopic velocity
        for x in range(nx + 2):
            for y in range(ny + 2):
                u = bulk_velocity[x, y]
                s = np.zeros(9, np.float32)

                for i in range(9):
                    e_dot_u = (u[0] * e[i, 0]) + (u[1] * e[i, 1])
                    s[i] = (
                        1
                        + (e_dot_u * 3)
                        + ((9 / 2.0) * np.square(e_dot_u))
                        - ((3 / 2.0) * (np.square(u[0] + np.square(u[1]))))
                    )

                eq_f = rho * w * s

                if init is True:
                    f[x, y] = eq_f
                else:
                    f[x, y] = (f[x, y] * (1 - (1 / relax_time))) + (eq_f / relax_time)

    def boundary_condition(self):
        # sets the preset boundary condition
        self.bulk_velocity[0, 2:-2] = [0.3, 0]
        # self.bulk_velocity[self.nx, :] = [0.9, 0]
        # self.bulk_velocity[self.nx+1, :] = [1., 0]

    def stream(self):
        self._stream(
            self.f,
            self.ny,
            self.nx,
            self.reflect,
            self.w,
            self.old_f,
            self.obs_location_list,
        )

    @staticmethod
    @jit(nopython=True)
    def _stream(f, ny, nx, reflect, w, old_f, obs_list):
        # reallocates the distribution function to new position, also deals with obstacle
        old_f = f.copy()
        # Horizontal

        f[:, :, 0] = old_f[:, :, 0]
        f[1 : nx + 2, :, 1] = old_f[0 : nx + 1, :, 1]
        f[:, 0 : ny + 1, 2] = old_f[:, 1 : ny + 2, 2]
        f[0 : nx + 1, :, 3] = old_f[1 : nx + 2, :, 3]
        f[:, 1 : ny + 2, 4] = old_f[:, 0 : ny + 1, 4]

        f[1 : nx + 2, 0 : ny + 1, 5] = old_f[0 : nx + 1, 1 : ny + 2, 5]
        f[0 : nx + 1, 0 : ny + 1, 6] = old_f[1 : nx + 2, 1 : ny + 2, 6]
        f[0 : nx + 1, 1 : ny + 2, 7] = old_f[1 : nx + 2, 0 : ny + 1, 7]
        f[1 : nx + 2, 1 : ny + 2, 8] = old_f[0 : nx + 1, 0 : ny + 1, 8]

        f[:, 0, reflect[2]] = old_f[:, 0, 2]
        f[:, 0, reflect[5]] = old_f[:, 0, 5]
        f[:, 0, reflect[6]] = old_f[:, 0, 6]

        f[:, ny + 1, reflect[4]] = old_f[:, ny + 1, 4]
        f[:, ny + 1, reflect[7]] = old_f[:, ny + 1, 7]
        f[:, ny + 1, reflect[8]] = old_f[:, ny + 1, 8]

        for x, y in obs_list:
            f[x + 1, y, 1] = f[x, y, reflect[1]]
            f[x, y + 1, 2] = f[x, y, reflect[2]]
            f[x - 1, y, 3] = f[x, y, reflect[3]]
            f[x, y - 1, 4] = f[x, y, reflect[4]]

            f[x + 1, y + 1, 5] = f[x, y, reflect[5]]
            f[x - 1, y + 1, 6] = f[x, y, reflect[6]]
            f[x - 1, y - 1, 7] = f[x, y, reflect[7]]
            f[x + 1, y - 1, 8] = f[x, y, reflect[8]]

            f[x, y] = w.copy()

    def __stream(self):
        # reallocates the distribution function to new position, also deals with obstacle
        old_f = self.f.copy()
        reflect = [0, 3, 4, 1, 2, 7, 8, 5, 6]
        # Horizontal

        self.f[:, :, 0] = old_f[:, :, 0]
        self.f[1 : self.nx + 2, :, 1] = old_f[0 : self.nx + 1, :, 1]
        self.f[:, 0 : self.ny + 1, 2] = old_f[:, 1 : self.ny + 2, 2]
        self.f[0 : self.nx + 1, :, 3] = old_f[1 : self.nx + 2, :, 3]
        self.f[:, 1 : self.ny + 2, 4] = old_f[:, 0 : self.ny + 1, 4]

        self.f[1 : self.nx + 2, 0 : self.ny + 1, 5] = old_f[
            0 : self.nx + 1, 1 : self.ny + 2, 5
        ]
        self.f[0 : self.nx + 1, 0 : self.ny + 1, 6] = old_f[
            1 : self.nx + 2, 1 : self.ny + 2, 6
        ]
        self.f[0 : self.nx + 1, 1 : self.ny + 2, 7] = old_f[
            1 : self.nx + 2, 0 : self.ny + 1, 7
        ]
        self.f[1 : self.nx + 2, 1 : self.ny + 2, 8] = old_f[
            0 : self.nx + 1, 0 : self.ny + 1, 8
        ]

        self.f[:, 0, reflect[2]] = old_f[:, 0, 2]
        self.f[:, 0, reflect[5]] = old_f[:, 0, 5]
        self.f[:, 0, reflect[6]] = old_f[:, 0, 6]

        self.f[:, self.ny + 1, reflect[4]] = old_f[:, self.ny + 1, 4]
        self.f[:, self.ny + 1, reflect[7]] = old_f[:, self.ny + 1, 7]
        self.f[:, self.ny + 1, reflect[8]] = old_f[:, self.ny + 1, 8]

        for x, y in self.obs_location_list:
            self.f[x + 1, y, 1] = self.f[x, y, reflect[1]]
            self.f[x, y + 1, 2] = self.f[x, y, reflect[2]]
            self.f[x - 1, y, 3] = self.f[x, y, reflect[3]]
            self.f[x, y - 1, 4] = self.f[x, y, reflect[4]]

            self.f[x + 1, y + 1, 5] = self.f[x, y, reflect[5]]
            self.f[x - 1, y + 1, 6] = self.f[x, y, reflect[6]]
            self.f[x - 1, y - 1, 7] = self.f[x, y, reflect[7]]
            self.f[x + 1, y - 1, 8] = self.f[x, y, reflect[8]]

            self.f[x, y] = self.w.copy()

    def set_f(self, old_f_val, tx, ty, i, fx, fy):
        if self.obstacle[tx, ty]:
            self.f[fx, fy, self.reflect[i]] = old_f_val

        else:
            self.f[tx, ty, i] = old_f_val

    def forcing_term(self):
        # adds the forcing term to the distribution function
        # used if there is a force pushing the fluid horizontally
        f_term = 8.0 * self.viscosity * 1 * self.rho / (6.0 * self.ny * self.ny)
        self.f[:, :, 1] += f_term
        self.f[:, :, 5] += f_term
        self.f[:, :, 8] += f_term

        self.f[:, :, 3] -= f_term
        self.f[:, :, 6] -= f_term
        self.f[:, :, 7] -= f_term

    def simulate_(self, nt):
        # runs the simulation for nt number of steps, where dt = 1
        for t in range(nt):
            self.stream()
            # self.refresh_values()
            self.new_velocity()
            self.boundary_condition()
            self.collision()
            self.forcing_term()

    def simulate_write(self, nt):
        # runs the simulation for nt number of steps, where dt = 1
        with open(self.file, "w+") as file:
            file.write(
                '[{"tao":'
                + str(self.relax_time)
                + ',"dimension":'
                + str(list(self.obstacle.shape))
                + ',"viscosity":'
                + str(self.viscosity)
                + ',"density":'
                + str(self.rho)
                + "}"
            )
            for t in range(nt):
                file.write(
                    ',\n{"' + str(t) + '":' + str(self.bulk_velocity.tolist()) + "}"
                )
                self.stream()
                self.new_velocity()
                self.boundary_condition()
                self.collision()

            file.write("]")

    def graph_rho(self):
        # represents density as a heatmap
        plt.imshow(self.rho, cmap="hot")
        plt.show()

    def graph_u_vector(self, return_val=False, ax=None):
        # graphs the microscopic velocity of the current time step
        # or initialises the vector field if return_val is True

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        x, y = np.meshgrid(
            np.linspace(0, self.nx, int(self.nx / 4) + 1),
            np.linspace(0, self.ny, int(self.ny / 4) + 1),
        )
        Q = ax.quiver(
            x,
            y,
            self.bulk_velocity[1:-1:4, 1:-1:4, 0].T,
            self.bulk_velocity[1:-1:4, 1:-1:4, 1].T,
            pivot="mid",
        )

        ax.imshow(~self.obstacle.T, cmap="gray", alpha=0.8)

        if return_val:
            return Q, x, y

        if ax is None:
            plt.show()

    def graph_heatmap(self, return_val=False, ax=None):
        # graphs the magnitude of microscopic velocity of current time step as heatmap
        # or initialise the heatmap if return_val is True

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        data = np.transpose(
            np.sqrt(self.bulk_velocity[:, :, 0] ** 2 + self.bulk_velocity[:, :, 1] ** 2)
        )

        Q = ax.imshow(data)
        del data

        if return_val:
            return Q

        if ax is None:
            plt.show()

    def update_heatmap(self, num, Q, x, y):
        self.simulate(1)
        data = np.transpose(
            np.sqrt(self.bulk_velocity[:, :, 0] ** 2 + self.bulk_velocity[:, :, 1] ** 2)
        )
        Q.set_data(data)
        return Q

    def update_vector(self, num, Q, x, y):
        # updates the vector in the vector field for animation
        self.simulate(1)
        Q.set_UVC(
            np.transpose(self.bulk_velocity[1 : self.nx + 1, 1 : self.ny + 1, 0]),
            np.transpose(self.bulk_velocity[1 : self.nx + 1, 1 : self.ny + 1, 1]),
        )
        return Q

    def graph_u_streamline(self, ax=None):
        # graphs the microscopic velocities as a streamline graph

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        X, Y = np.meshgrid(
            np.linspace(0, self.nx, self.nx),
            np.linspace(0, self.ny, self.ny),
        )
        ax.streamplot(
            X,
            Y,
            self.bulk_velocity[1:-1, 1:-1, 0].T,
            self.bulk_velocity[1:-1, 1:-1, 1].T,
        )

        ax.imshow(~self.obstacle[1:-1, 1:-1].T, cmap="gray", alpha=0.8)

        if ax is None:
            plt.show()

    def graph_u_contour(self, ax=None):
        # graphs the microscopic velocities as a contour graph

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        X, Y = np.meshgrid(
            np.linspace(0, self.nx, self.nx),
            np.linspace(0, self.ny, self.ny),
        )

        ax.contourf(
            X,
            Y,
            np.transpose(
                np.sqrt(
                    np.square(self.bulk_velocity[1:-1, 1:-1, 1])
                    + np.square(self.bulk_velocity[1:-1, 1:-1, 0])
                )
            ),
        )

        ax.imshow(~self.obstacle[1:-1, 1:-1].T, cmap="gray", alpha=0.8)

        if ax is None:
            plt.show()

    def set_obstacle(self, bool_matrix):
        # sets the obstacle with boolean matrix
        self.obstacle[1:-1, 1:-1] = bool_matrix.copy()
        self.obs_location_list = []

        for x in range(self.nx):
            for y in range(self.ny):
                # print(type(self .obstacle[x, y]))
                if self.obstacle[x, y]:
                    self.obs_location_list.append((x, y))

        self.obs_location_list = List(self.obs_location_list)

    def set_tao(self, tao):
        # sets relaxation time and redefines viscosity accordingly, dt = 1
        self.relax_time = tao
        self.viscosity = ((2 * tao) - 1) / 6

    def set_viscosity(self, viscosity):
        self.viscosity = viscosity
        self.relax_time = ((6 * viscosity) + 1) / 2

    def save_to(self, file_path="temp.json"):
        self.file = file_path
        self.simulate = self.simulate_write

    def start_with_all(self):
        self.bulk_velocity[:, :] = [0.5, 0.0]

        for x, y in self.obs_location_list:
            self.bulk_velocity[x, y] = [0.0, 0.0]


def create_exapmple_model() -> Lattice:
    # creates a 50*70 model with an obstacle

    model = Lattice(70, 50)
    model.set_viscosity(1 / 6)
    matrix = np.zeros((70, 50), bool)
    # matrix[:, 0] = True
    # matrix[1, :] = True
    # matrix[20, :] = True
    # matrix[:, 51] = True
    matrix[8:12, 20:30] = True
    # matrix[191, :] = True

    model.set_obstacle(matrix)
    return model


def plot_graphs(model: Lattice):
    # plots the graphs after 500 steps of simulation

    model.simulate(500)

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    print(ax.shape)
    model.graph_u_streamline(ax=ax[0, 0])
    model.graph_u_contour(ax=ax[0, 1])
    model.graph_heatmap(ax=ax[1, 0])
    model.graph_u_vector(ax=ax[1, 1])

    plt.show()


def vector_animation(model: Lattice):
    # runs the vector field as an animation

    fig = plt.figure(figsize=(7, 5))
    q, x, y = model.graph_u_vector(return_val=True)
    anim = animation.FuncAnimation(
        fig, model.update_vector, fargs=(q, x, y), interval=50, blit=False
    )
    plt.show()


def heatmap_animation(model: Lattice):
    # runs the heatmap as an animation

    fig = plt.figure(figsize=(7, 5))
    p = model.graph_heatmap(return_val=True)
    x = 1
    y = 2

    anim2 = animation.FuncAnimation(
        fig, model.update_heatmap, fargs=(p, x, y), interval=50, blit=False
    )
    plt.show()


if __name__ == "__main__":
    # help(__import__(__name__))

    model = create_exapmple_model()

    plot_graphs(model)

    # vector_animation(model)

    # heatmap_animation(model)
