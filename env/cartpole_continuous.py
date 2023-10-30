"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class CartPoleContinuousEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)  # 0.1+1.0=1.1
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)  # 0.1*0.5=0.05
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed()

        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        self.phase = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg

        # x, x_dot, theta, theta_dot = self.state
        # # action = np.clip(action, -5, 5)
        # # force = self.force_mag if action == 1 else -self.force_mag
        # force = action
        # costheta = math.cos(theta)
        # sintheta = math.sin(theta)
        #
        # # "temp=(a+0.05*x4*x4*sin(x3))/1.1",
        # # "tehtaacc=(9.8*sin(x3)-cos(x3)*((a+0.05*x4*x4*sin(x3))/1.1))/(0.5*(4.0/3.0-(0.1*cos(x3)*cos(x3)/1.1)))",
        # # "xacc = ((a+0.05*x4*x4*sin(x3))/1.1) - (0.05 * ((9.8*sin(x3)-cos(x3)*((a+0.05*x4*x4*sin(x3))/1.1))/(0.5*(4.0/3.0-(0.1*cos(x3)*cos(x3)/1.1))))*cos(x3))/1.1",
        # # For the interested reader:
        # # https://coneural.org/florian/papers/05_cart_pole.pdf
        # temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        # thetaacc = (self.gravity * sintheta - costheta * temp) / (
        #         self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        # xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        #
        # x = x + self.tau * x_dot
        # x_dot = x_dot + self.tau * xacc
        # theta = theta + self.tau * theta_dot
        # theta_dot = theta_dot + self.tau * thetaacc
        #
        # self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        self.step_size(action)
        # self.state = np.array([x1_new, x2_new, x3_new], dtype=np.float32)
        x, x_dot, theta, theta_dot = self.state
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
            reward = 2 - x * x - 10 * abs(theta)
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = -200
            # reward =0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = -200
            # reward=0

        return self._get_obs(), reward, done, {}

    def step_size(self,u, step_size=0.005):
        done = False
        # u = np.clip(u, -self.th, self.th)[0]

        offset = 0
        scala = 1
        u = u[0] - offset
        u = u * scala
        t = self.tau
        time = 0
        state_list = []
        while time < t:
            x, x_dot, theta, theta_dot = self.state
            # action = np.clip(action, -5, 5)
            # force = self.force_mag if action == 1 else -self.force_mag
            force = u
            costheta = math.cos(theta)
            sintheta = math.sin(theta)

            # "temp=(a+0.05*x4*x4*sin(x3))/1.1",
            # "tehtaacc=(9.8*sin(x3)-cos(x3)*((a+0.05*x4*x4*sin(x3))/1.1))/(0.5*(4.0/3.0-(0.1*cos(x3)*cos(x3)/1.1)))",
            # "xacc = ((a+0.05*x4*x4*sin(x3))/1.1) - (0.05 * ((9.8*sin(x3)-cos(x3)*((a+0.05*x4*x4*sin(x3))/1.1))/(0.5*(4.0/3.0-(0.1*cos(x3)*cos(x3)/1.1))))*cos(x3))/1.1",
            # For the interested reader:
            # https://coneural.org/florian/papers/05_cart_pole.pdf
            temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

            x = x + step_size * x_dot
            x_dot = x_dot + step_size * xacc
            theta = theta + step_size * theta_dot
            theta_dot = theta_dot + step_size * thetaacc

            state_list.append([x, theta])
            self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
            time = round(time + step_size, 10)
        return self.state, state_list, done

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        high = np.array([0.02, 0.02, 0.021, 0.02])
        low = np.array([0.02, 0.02, 0.02, 0.02])
        self.state = self.np_random.uniform(low=low, high=high)
        self.steps_beyond_done = None
        return np.array(self.state)

    def _get_obs(self):
        # self.state[0] = np.clip(self.state[0], -5, 5)
        # self.state[1] = np.clip(self.state[1], -10, 10)
        # self.state[2] = np.clip(self.state[2], -0.5, 0.5)
        # self.state[3] = np.clip(self.state[3], -10, 10)
        self.state[0] = np.clip(self.state[0], -2.5, 2.5)
        self.state[1] = np.clip(self.state[1], -10, 10)
        self.state[2] = np.clip(self.state[2], -0.25, 0.25)
        self.state[3] = np.clip(self.state[3], -10, 10)
        return self.state

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def set_phase(self, phase):
        self.phase = phase

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
