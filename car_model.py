import casadi as ca
from casadi import exp, sin, cos, arctan
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


class CarModel:
    def __init__(self):
        # define states
        states = ca.MX.sym("states", 7)
        x, y, v, beta, psi, wz, delta = (
            states[0],
            states[1],
            states[2],
            states[3],
            states[4],
            states[5],
            states[6],
        )

        # define controls
        controls = ca.MX.sym("controls", 4)
        wdelta = controls[0]
        brake_force = controls[1]
        accelerator = controls[2]
        gear = controls[3]

        # parameters
        m = 1239
        g = 9.81
        lf = 1.19016
        lr = 1.37484
        eSP = 0.5
        R = 0.302
        Izz = 1752
        cw = 0.3
        rho = 1.249512
        A = 1.4378946874
        ig1, ig2, ig3, ig4, ig5 = 3.91, 2.002, 1.33, 1.0, 0.805
        it = 3.91
        Bf, Br, Cf, Cr, Df, Dr, Ef, Er = (
            10.96,
            12.67,
            1.3,
            1.3,
            4560.40,
            3947.81,
            -0.5,
            -0.5,
        )
        fR0, fR1, fR4 = 0.009, 0.002, 0.0003
        B = 1.5  # car width
        # Aerodynamic forces
        FAx = 0.5 * cw * rho * A * v**2
        FAy = 0.0
        ig_mu = ca.if_else(
            gear < 1.5,
            ig1,
            ca.if_else(
                gear < 2.5,
                ig2,
                ca.if_else(gear < 3.5, ig3, ca.if_else(gear < 4.5, ig4, ig5)),
            ),
        )

        dpsidt = wz

        # calculate slip angles
        alphaf = delta - arctan((lf * dpsidt - v * sin(beta)) / (v * cos(beta)))
        alphar = arctan((lr * dpsidt + v * sin(beta)) / (v * cos(beta)))
        # calculate lateral tire forces wear
        # Fsf depends on alphaf, Fsr depends on alphar
        Fsf = Df * sin(
            Cf * arctan(Bf * alphaf - Ef * (Bf * alphaf - arctan(Bf * alphaf)))
        )
        Fsr = Dr * sin(
            Cr * arctan(Br * alphar - Er * (Br * alphar - arctan(Br * alphar)))
        )

        # static tire loads at the front and rear wheel
        Fzf = m * lr * g / (lf + lr)
        Fzr = m * lf * g / (lf + lr)

        # front, rear breaking forces
        FBf = 2 / 3 * brake_force
        FBr = 1 / 3 * brake_force

        # friction coefficient
        fRv = fR0 + fR1 * v / 100 + fR4 * (v / 100) ** 4

        # rolling resistance forces
        FRf = fRv * Fzf
        FRr = fRv * Fzr

        # motor torque
        wmot = v * ig_mu * it / R

        f1 = 1 - exp(-3 * accelerator)
        f2 = -37.8 + 1.54 * wmot - 0.0019 * wmot**2
        f3 = -34.9 - 0.04775 * wmot

        M_mot = f1 * f2 + (1.0 - f1) * f3
        M_wheel = ig_mu * it * M_mot
        Flr = M_wheel / R - FBr - FRr
        Flf = -FBf - FRf

        dxdt = v * cos(psi - beta)
        dydt = v * sin(psi - beta)
        dvdt = (
            1
            / m
            * (
                (Flr - FAx) * cos(beta)
                + Flf * cos(delta + beta)
                - (Fsr - FAy) * sin(beta)
                - Fsf * sin(delta + beta)
            )
        )
        dbetadt = wz - 1 / (m * v) * (
            (Flr - FAx) * sin(beta)
            + Flf * sin(delta + beta)
            + (Fsr - FAy) * cos(beta)
            + Fsf * cos(delta + beta)
        )
        dwzdt = (
            1
            / Izz
            * (Fsf * lf * cos(delta) - Fsr * lr - FAy * eSP + Flf * lf * sin(delta))
        )
        ddeltadt = wdelta

        self.car_dynamics = ca.vertcat(
            dxdt, dydt, dvdt, dbetadt, dpsidt, dwzdt, ddeltadt
        )

        self.controls = controls
        self.states = states
        self.B = B
        self.L = 2 # example length of car

    def get_system(self):
        """
        Gives car model system

        :return states:
        :return controls:
        :return car_dynamics:
        """
        return self.states, self.controls, self.car_dynamics

    def trajectory(self, controls, x0, t_grid):
        """
        Compute trajectory of car

        :param controls: controls (in this order!) steering angle velocity, breaking force, breaking_force, acceleration, gear
        :param x0: initial states (in this order!) x0, y0, v0, beta0, psi0, wz0, delta0
        :param t_grid: time grid

        :return trajectory: np.array with states at one point on t_grid in each line
        """
        trajectory = []
        xk = x0
        system = {"ode": self.car_dynamics, "x": self.states, "p": self.controls}
        if self.is_uniform_grid(t_grid):
            # if the grid is uniform, we only need one integrator and can save time and storage
            integrator = ca.integrator(
                f"integrator", "cvodes", system, 0, t_grid[1] - t_grid[0]
            )
            for i in range(t_grid.size - 1):
                res = integrator(x0=xk, p=controls[i])
                xk = res["xf"]
                trajectory.append(xk)

        else:
            for i in range(t_grid.size - 1):
                # for not uniform time grid
                t0, t1 = t_grid[i], t_grid[i + 1]
                integrator = ca.integrator(f"integrator_{i}", "cvodes", system, t0, t1)
                res = integrator(x0=xk, p=controls[i])
                xk = res["xf"]
                trajectory.append(xk)

        return np.array([np.array(x).squeeze() for x in trajectory])

    def smoothed_trajectory(
        self, controls, x0, t_grid, refinement=10, return_grid=False
    ):
        """
        Compute smoothed trajectory of car

        :param (np.ndarray) controls: controls (in this order!) steering angle velocity, breaking force, breaking_force, acceleration, gear
        :param (np.ndarray) x0: initial states (in this order!) x0, y0, v0, beta0, psi0, wz0, delta0
        :param (np.ndarray) t_grid: time grid
        :param (int) refinement: number of additional integration nodes per interval
        :param return_grid: return refined grid as well

        :return (optional) grid: refined grid
        :return trajectory: np.array with states at one point on the refined grid in each line
        """
        refined_grid = self.refine_grid(t_grid, refinement)
        trajectory = [x0]
        xk = x0
        system = {"ode": self.car_dynamics, "x": self.states, "p": self.controls}
        control_idx = (
            0  # we need this so that we dont have to create a much bigger control array
        )
        if self.is_uniform_grid(refined_grid):
            # if the grid is uniform, we only need one integrator and can save time and storage
            integrator = ca.integrator(
                f"integrator", "cvodes", system, 0, refined_grid[1] - refined_grid[0]
            )
            for grid_idx in range(refined_grid.size - 1):
                res = integrator(x0=xk, p=controls[control_idx])
                xk = res["xf"]
                trajectory.append(xk)
                if grid_idx % refinement == 0 and grid_idx != 0:
                    control_idx += 1

        else:
            for i in range(refined_grid.size - 1):
                # for not uniform time grid
                t0, t1 = refined_grid[i], refined_grid[i + 1]
                integrator = ca.integrator(f"integrator_{i}", "cvodes", system, t0, t1)
                res = integrator(x0=xk, p=controls[control_idx])
                xk = res["xf"]
                trajectory.append(xk)
                if grid_idx % refinement == 0 and grid_idx != 0:
                    control_idx += 1

        traj = np.array([np.array(x).squeeze() for x in trajectory])
        if return_grid:
            return refined_grid, traj
        else:
            return traj

    def animate(
        self, controls, x0, t_grid, speedup=1, smoothing=1, filename="car_animation.mp4"
    ):
        """
        Animate car trajectory given controls

        :param (np.ndarray) controls: controls (in this order!) steering angle velocity, breaking force, breaking_force, acceleration, gear
        :param (np.ndarray) x0: initial states (in this order!) x0, y0, v0, beta0, psi0, wz0, delta0
        :param (np.ndarray) t_grid: time grid
        :param (float) speedup: factor to speed video up by 
        :param (int) smoothing: smoothes the trajectory by decreasing the size of the integration interval
        :param (string) filename: filename for rendered video (must end in .mp4!)
        """
        self.L = 2
        if smoothing == 1:
            trajectory = self.trajectory(controls, x0, t_grid)
        else:
            t_grid, trajectory = self.smoothed_trajectory(
                controls, x0, t_grid, refinement=smoothing, return_grid=True
            )
        fig, ax = plt.subplots()
        x_max, x_min = np.max(trajectory[:, 0]), np.min(trajectory[:, 0])
        y_max, y_min = np.max(trajectory[:, 1]), np.min(trajectory[:, 1])
        ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
        enhancement = (
            max(x_max - x_min, y_max - y_min) / self.L * 0.03
        )  # make car size 3% of the track
        line = ax.plot(trajectory[0, 0], trajectory[0, 1], linestyle="--")[0]
        car = plt.Rectangle(
            (
                trajectory[0, 0] - (enhancement * self.L / 2),
                trajectory[0, 1] - (enhancement * self.B / 2),
            ),
            enhancement * self.L,
            enhancement * self.B,
            angle=trajectory[0, 4],
            rotation_point="center",
            color="k",
            fill=True,
        )

        def update(frame):
            x_until_now = trajectory[: frame + 1, 0]
            y_until_now = trajectory[: frame + 1, 1]
            x_now = trajectory[frame, 0]
            y_now = trajectory[frame, 1]
            angle_now = np.rad2deg(trajectory[frame, 4])
            car.set_x(x_now - (enhancement * self.L / 2))
            car.set_y(y_now - (enhancement * self.B / 2))
            car.set_angle(angle_now)
            line.set_xdata(x_until_now)
            line.set_ydata(y_until_now)
            ax.add_patch(car)
            ax.set_title(
                f"Time: {round(t_grid[frame],3)}s - Velocity {round(trajectory[frame,2],3)} m/s",
            )
            return (line,)

        dt = np.diff(t_grid, prepend=t_grid[0])
        avg_dt_ms = np.mean(dt) * 1000 / speedup
        fps = 1000 / avg_dt_ms

        ani = animation.FuncAnimation(
            fig, update, frames=len(t_grid), interval=avg_dt_ms, blit=True
        )
        ani.save(filename, writer="ffmpeg", fps=fps)

    @staticmethod
    def is_uniform_grid(grid):
        diffs = grid[1:] - grid[:-1]
        return np.allclose(diffs, diffs[0], rtol=1e-8)

    @staticmethod
    def refine_grid(t_grid, refinement=2):
        refined_grid = []
        for i in range(t_grid.size - 1):
            t0, t1 = t_grid[i], t_grid[i + 1]
            refined_grid.append(np.linspace(t0, t1, refinement + 1)[:-1])
        refined_grid.append([t_grid[-1]])
        refined_grid = np.concatenate(refined_grid)
        return refined_grid
