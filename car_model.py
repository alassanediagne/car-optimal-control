import casadi as ca
from casadi import exp, sin, cos, arctan
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.interpolate import interp1d
from matplotlib.patches import Rectangle


class CarModel:
    def __init__(self):
        # final time
        tf = ca.MX.sym("tf")
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
            tf * dxdt,
            tf * dydt,
            tf * dvdt,
            tf * dbetadt,
            tf * dpsidt,
            tf * dwzdt,
            tf * ddeltadt,
        )
        # multiply with tf to normalize time

        self.controls = controls
        self.states = states
        self.tf = tf
        self.B = B
        self.L = 2  # example length of car

    def get_system(self):
        """
        Gives car model system

        :return final_time:
        :return states: x, y, v, beta, psi, wz, delta
        :return controls: steering angle velocity, breaking force, acceleration, gear
        :return car_dynamics:
        """
        return self.tf, self.states, self.controls, self.car_dynamics

    def trajectory(self, controls, x0, t_grid, final_time):
        """
        Compute trajectory of car

        :param controls: controls (in this order!) steering angle velocity, breaking force, acceleration, gear
        :param x0: initial states (in this order!) x0, y0, v0, beta0, psi0, wz0, delta0
        :param t_grid: time grid

        :return trajectory: np.array with states at one point on t_grid in each line
        """
        trajectory = []
        xk = x0
        system = {"ode": self.car_dynamics, "x": self.states, "p": ca.vertcat(self.tf, self.controls)}
        if self.is_uniform_grid(t_grid):
            # if the grid is uniform, we only need one integrator and can save time and storage
            integrator = ca.integrator(
                f"integrator", "cvodes", system, 0, t_grid[1] - t_grid[0]
            )
            for i in range(t_grid.size - 1):
                res = integrator(x0=xk, p=np.concatenate((final_time,controls[i])))
                xk = res["xf"]
                trajectory.append(xk)

        else:
            for i in range(t_grid.size - 1):
                # for not uniform time grid
                t0, t1 = t_grid[i], t_grid[i + 1]
                integrator = ca.integrator(f"integrator_{i}", "cvodes", system, t0, t1)
                res = integrator(x0=xk, p=np.concatenate((final_time,controls[i])))
                xk = res["xf"]
                trajectory.append(xk)

        return np.array([np.array(x).squeeze() for x in trajectory])

    def smoothed_trajectory(
        self, controls, x0, t_grid, final_time, refinement=10, return_grid=False
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
        system = {"ode": self.car_dynamics, "x": self.states, "p": ca.vertcat(self.tf, self.controls)}
        control_idx = (
            0  # we need this so that we dont have to create a much bigger control array
        )
        if self.is_uniform_grid(refined_grid):
            # if the grid is uniform, we only need one integrator and can save time and storage
            integrator = ca.integrator(
                f"integrator", "cvodes", system, 0, refined_grid[1] - refined_grid[0]
            )
            for grid_idx in range(refined_grid.size - 1):
                res = integrator(x0=xk, p=np.concatenate(([final_time], controls[control_idx])))
                xk = res["xf"]
                trajectory.append(xk)
                if grid_idx % refinement == 0 and grid_idx != 0:
                    control_idx += 1

        else:
            for grid_idx in range(refined_grid.size - 1):
                # for not uniform time grid
                t0, t1 = refined_grid[i], refined_grid[i + 1]
                integrator = ca.integrator(f"integrator_{i}", "cvodes", system, t0, t1)
                res = integrator(x0=xk, p=np.concatenate(([final_time], controls[control_idx])))
                xk = res["xf"]
                trajectory.append(xk)
                if grid_idx % refinement == 0 and grid_idx != 0:
                    control_idx += 1

        traj = np.array([np.array(x).squeeze() for x in trajectory])
        if return_grid:
            return refined_grid, traj
        else:
            return traj

    def plot_trajectory(
        self,
        controls,
        x0,
        t_grid,
        final_time,
        track_params=None,
        track_limits=None,
        smoothing=1,
        filename=None,
        title = None
    ):
        if track_params is not None:
            assert track_limits is not None, "boundary limits needed"
            Pl, Pu = self.make_track(self.states[0],*track_params)
            xs = np.linspace(*track_limits, 500)
            Pl_func = ca.Function("Pl", [self.states], [Pl])
            Pu_func = ca.Function("Pu", [self.states], [Pu])
            track_upper = [Pu_func(xi).full().item() for xi in xs]
            track_lower = [Pl_func(xi).full().item() for xi in xs]
        self.L = 2
        if smoothing == 1:
            trajectory = self.trajectory(controls, x0, t_grid, final_time)
        else:
            t_grid, trajectory = self.smoothed_trajectory(
                controls, x0, t_grid, final_time, refinement=smoothing, return_grid=True
            )
        fig, ax = plt.subplots()
        if track_params is None:
            x_max, x_min = np.max(trajectory[:, 0]), np.min(trajectory[:, 0])
            y_max, y_min = np.max(trajectory[:, 1]), np.min(trajectory[:, 1])
        else:
            x_min, x_max = track_limits
            y_max, y_min = np.max(track_upper) + 1, np.min(track_lower) - 1
        ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="r")
        if track_params is not None:
            ax.plot(xs, track_lower, color="k")
            ax.plot(xs, track_upper, color="k")
        if title is not None:
            ax.set_title(title)
        if filename is not None:
            fig.savefig(filename)
        else:
            plt.show()

    def animate(
        self,
        controls,
        x0,
        t_grid,
        final_time,
        track_params=None,
        track_limits=None,
        speedup=1,
        smoothing=1,
        filename="car_animation.mp4",
        title = None
    ):
        """
        Animate car trajectory given controls

        :param (np.ndarray) controls: controls (in this order!) steering angle velocity, breaking force, breaking_force, acceleration, gear
        :param (np.ndarray) x0: initial states (in this order!) x0, y0, v0, beta0, psi0, wz0, delta0
        :param (np.ndarray) t_grid: time grid
        :param (tuple) track_params: h1,h2,h3,h4 that define the track
        :param (tuple) track_limits: limits of the track on the horizontal axis
        :param (float) speedup: factor to speed video up by
        :param (int) smoothing: smoothes the trajectory by decreasing the size of the integration interval
        :param (string) filename: filename for rendered video (must end in .mp4!)
        """
        if track_params is not None:
            assert track_limits is not None, "boundary limits needed"
            Pl, Pu = self.make_track(self.states[0],*track_params)
            xs = np.linspace(*track_limits, 500)
            Pl_func = ca.Function("Pl", [self.states], [Pl])
            Pu_func = ca.Function("Pu", [self.states], [Pu])
            track_upper = [Pu_func(xi).full().item() for xi in xs]
            track_lower = [Pl_func(xi).full().item() for xi in xs]

        self.L = 2
        if smoothing == 1:
            trajectory = self.trajectory(controls, x0, t_grid)
        else:
            t_grid, trajectory = self.smoothed_trajectory(
                controls, x0, t_grid, final_time, refinement=smoothing, return_grid=True
            )
        fig, ax = plt.subplots()
        if track_params is None:
            x_max, x_min = np.max(trajectory[:, 0]), np.min(trajectory[:, 0])
            y_max, y_min = np.max(trajectory[:, 1]), np.min(trajectory[:, 1])
        else:
            x_min, x_max = track_limits
            y_max, y_min = np.max(track_upper) + 1, np.min(track_lower) - 1
        ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

        # make car reasonably sized, independent of x and y axis
        car_ratio = self.B / self.L
        frame_width = x_max - x_min
        frame_height = y_max - y_min
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_window_extent(renderer=renderer)
        display_ratio = (bbox.width / bbox.height) * (frame_height / frame_width)
        car_length = 0.05 * frame_width
        car_width = car_ratio * car_length * display_ratio

        ax.plot(trajectory[:, 0], trajectory[:, 1], linestyle="--", color="gray")
        if track_params is not None:
            ax.plot(xs, track_lower, color="k")
            ax.plot(xs, track_upper, color="k")
        line = ax.plot(trajectory[0, 0], trajectory[0, 1], color="r")[0]
        if title is None:
            title = " "
        car = Rectangle(
            (
                trajectory[0, 0] - (car_length / 2),
                trajectory[0, 1] - (car_width / 2),
            ),
            car_length,
            car_width,
            angle=trajectory[0, 4],
            rotation_point="center",
            color="k",
            fill=True,
        )
        ax.add_patch(car)

        def update(frame):
            x_until_now = trajectory[: frame + 1, 0]
            y_until_now = trajectory[: frame + 1, 1]
            x_now = trajectory[frame, 0]
            y_now = trajectory[frame, 1]
            angle_now = np.rad2deg(trajectory[frame, 4])
            car.set_x(x_now - (car_length / 2))
            car.set_y(y_now - (car_width / 2))
            car.set_angle(angle_now)
            line.set_xdata(x_until_now)
            line.set_ydata(y_until_now)
            ax.set_title(
                title + f"  Time: {round(final_time*t_grid[frame],3)}s - Velocity {round(trajectory[frame,2],3)} m/s",
            )
            return (line,)

        dt = np.diff(t_grid, prepend=t_grid[0])
        avg_dt_ms = np.mean(dt) * 1000 * final_time / speedup
        fps = 1000 / avg_dt_ms

        ani = animation.FuncAnimation(
            fig, update, frames=len(t_grid), interval=avg_dt_ms, blit=True
        )
        ani.save(filename, writer="ffmpeg", fps=fps)

    def animate_race(
        self,
        controls,
        x0,
        t_grids,
        final_times,
        labels,
        track_params=None,
        track_limits=None,
        speedup=1,
        smoothing=1,
        filename="car_animation.mp4",
        title = None
    ):
        """
        Animate a race between a number of cars
        :param (list) controls: list of controls
        :param (list) x0: list of starting points
        :param (list) t_grids: list of time grids
        :param (list) final_times: list of final times
        :param (list) labels: list of labels
        """
        assert len(controls) == len(labels)

        assert track_limits is not None and track_params is not None, "need track to race"
        Pl, Pu = self.make_track(self.states[0],*track_params)
        xs = np.linspace(*track_limits, 500)
        Pl_func = ca.Function("Pl", [self.states], [Pl])
        Pu_func = ca.Function("Pu", [self.states], [Pu])
        track_upper = [Pu_func(xi).full().item() for xi in xs]
        track_lower = [Pl_func(xi).full().item() for xi in xs]

        self.L = 2

        # get trajectories for all cars
        raw_t_grids, raw_trajectories = [], []

        for controls_, t_grid_, final_time_, x0_ in zip(controls, t_grids, final_times, x0):
            if smoothing == 1:
                traj = self.trajectory(controls_, x0_, t_grid_)
                t_real = t_grid_ * final_time_
            else:
                t_grid_refined, traj = self.smoothed_trajectory(
                    controls_, x0_, t_grid_, final_time_, refinement=smoothing, return_grid=True
                )
                t_real = t_grid_refined * final_time_
            raw_t_grids.append(t_real)
            raw_trajectories.append(traj)

        # use longest time grid as common grid
        common_t = max(raw_t_grids, key=len)

        # resample all trajectories to common time grid
        trajectories = []
        for t_src, traj in zip(raw_t_grids, raw_trajectories):
            new_traj = np.zeros((len(common_t), traj.shape[1]))
            for i in range(traj.shape[1]):
                f = interp1d(t_src, traj[:, i], kind='linear', fill_value="extrapolate")
                new_traj[:, i] = f(common_t)
            trajectories.append(new_traj)
    
        fig, ax = plt.subplots()
        x_min, x_max = track_limits
        y_max, y_min = np.max(track_upper) + 1, np.min(track_lower) - 1
        ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

        # make car reasonably sized, independent of x and y axis
        car_ratio = self.B / self.L
        frame_width = x_max - x_min
        frame_height = y_max - y_min
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_window_extent(renderer=renderer)
        display_ratio = (bbox.width / bbox.height) * (frame_height / frame_width)
        car_length = 0.05 * frame_width
        car_width = car_ratio * car_length * display_ratio
        lines, cars = [], []
        colors = plt.cm.rainbow(np.linspace(0, 1, len(labels)))

        for trajectory, label, c in zip(trajectories, labels, colors):
            ax.plot(trajectory[:, 0], trajectory[:, 1], linestyle="--", color="gray")
            l = ax.plot(trajectory[0, 0], trajectory[0, 1], color=c, label = label)[0]
            lines.append(l)
            cars.append(plt.Rectangle(
            (
                trajectory[0, 0] - (car_length / 2),
                trajectory[0, 1] - (car_width / 2),
            ),
            car_length,
            car_width,
            angle=trajectory[0, 4],
            rotation_point="center",
            color="k",
            fill=True,
        ))

        ax.plot(xs, track_lower, color="k")
        ax.plot(xs, track_upper, color="k")
        for car in cars:
            ax.add_patch(car)
        if title is None:
            title = " "

        def update(frame):
            for trajectory, car, line in zip(trajectories, cars, lines):
                x_until_now = trajectory[: frame + 1, 0]
                y_until_now = trajectory[: frame + 1, 1]
                x_now = trajectory[frame, 0]
                y_now = trajectory[frame, 1]
                angle_now = np.rad2deg(trajectory[frame, 4])
                car.set_x(x_now - (car_length / 2))
                car.set_y(y_now - (car_width / 2))
                car.set_angle(angle_now)
                line.set_xdata(x_until_now)
                line.set_ydata(y_until_now)
            ax.set_title(
                    title + f"  Time: {round(common_t[frame],3)}s",
               )
            return lines + cars
        ax.legend()
        avg_dt_ms = np.mean(np.diff(common_t, prepend=common_t[0])) * 1000 / speedup

        fps = 1000 / avg_dt_ms

        ani = animation.FuncAnimation(
            fig, update, frames=len(common_t), interval=avg_dt_ms, blit=True
        )
        ani.save(filename, writer="ffmpeg", fps=fps)
        

    def make_track(self, x, h1, h2, h3, h4):
        Pl = ca.if_else(
            x <= 44,
            0,
            ca.if_else(
                x > 71,
                0,
                ca.if_else(
                    x <= 44.5,
                    4 * h2 * (x - 44) ** 3,
                    ca.if_else(
                        x <= 45,
                        4 * h2 * (x - 45) ** 3 + h2,
                        ca.if_else(
                            x <= 70,
                            h2,
                            ca.if_else(
                                x <= 70.5,
                                4 * h2 * (70 - x) ** 3 + h2,
                                4 * h2 * (71 - x) ** 3,
                            ),
                        ),
                    ),
                ),
            ),
        )

        Pu = ca.if_else(
            x <= 15,
            h1,
            ca.if_else(
                x <= 15.5,
                4 * (h3 - h1) * (x - 15) ** 3 + h1,
                ca.if_else(
                    x <= 16,
                    4 * (h3 - h1) * (x - 16) ** 3 + h3,
                    ca.if_else(
                        x <= 94,
                        h3,
                        ca.if_else(
                            x <= 94.5,
                            4 * (h3 - h4) * (94 - x) ** 3 + h3,
                            ca.if_else(x <= 95, 4 * (h3 - h4) * (95 - x) ** 3 + h4, h4),
                        ),
                    ),
                ),
            ),
        )

        return (Pl, Pu)

    def get_state_bounds(self):
        lbx = -1 * ca.inf
        ubx = ca.inf
        lby = - ca.inf # set y boundaries by inequality constraints
        uby = ca.inf
        lbv = 0
        ubv = ca.inf  # theoretically :)
        lb_beta = -1 * ca.inf
        ub_beta = ca.inf
        lb_psi = -1 * ca.inf
        ub_psi = ca.inf
        lb_wz = -1 * ca.inf
        ub_wz = ca.inf
        lb_delta = -1 * ca.inf
        ub_delta = ca.inf
        lb_states = [lbx, lby, lbv, lb_beta, lb_psi, lb_wz, lb_delta]
        ub_states = [ubx, uby, ubv, ub_beta, ub_psi, ub_wz, ub_delta]
        return (lb_states, ub_states)

    def get_controls_bounds(self, gear):
        """
        We consider a fixed gear problem for simplicity
        """
        lb_steer_angle = -0.5
        ub_steer_angle = 0.5
        lb_break = 0
        ub_break = 1.5 * 1e4
        lb_acc = 0
        ub_acc = 1
        lbu = [lb_steer_angle, lb_break, lb_acc, gear]
        ubu = [ub_steer_angle, ub_break, ub_acc, gear]
        return (lbu, ubu)

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
