import gym
import numpy as np
from gym import (spaces, utils, logger)
from collections import OrderedDict
from math import floor
from six import StringIO
import torch
import sys


left_dir = {
    (0, 2): (-2, 0),
    (0, -2): (2, 0),
    (2, 0): (0, 2),
    (-2, 0): (0, -2),
}

p_to_int = {
    'H': 0,
    'P': 1
}

ACTION_TO_STR = {
    0 : 'F', 1 : 'L',
    2 : 'R'
}


class foldenv(gym.Env):
    def __init__(self, seq, collision_penalty=-2, trap_penalty=0.5):
        self.seq = seq
        self.collision_penalty = collision_penalty
        self.trap_penalty = trap_penalty

        # self.grid_length = 4 * len(seq) - 3
        self.grid_length = 63
        # 4 dimensions: H, P, primary connect, H-H connect
        self.grid = np.zeros((4, self.grid_length, self.grid_length), dtype=int)
        self.midpoint = (int(self.grid_length / 2), int(self.grid_length / 2))

        if seq[0] == 'H':
            self.grid[0][self.midpoint] = 1
        else:
            self.grid[1][self.midpoint] = 1

        self.state = OrderedDict({self.midpoint: seq[0]})

        # 0-forward, 1-left, 2-right
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.grid.shape, dtype=int)
        self.last_action = None
        self.last_action_dir = (-2, 0)
        self.actions = []

        self.collisions = 0
        self.trapped = 0

    def step(self, action):
        collision, is_trapped = False, False
        stone_idx = len(self.state)

        x, y = next(reversed(self.state))
        possible_dest = self._get_possible_dest((x, y))
        action_dest = possible_dest[action]

        # determine whether collide
        if self.grid[0][action_dest] == 1 or self.grid[1][action_dest] == 1:
            collision = True
            self.collisions += 1

        # determine whether trapped
        ind = 0
        tot_ind = 0
        for tu in possible_dest:
            if not (0 <= tu[0] <= self.grid_length - 1 and 0 <= tu[1] <= self.grid_length - 1):
                continue
            tot_ind += 1
            ind += self.grid[0][tu] + self.grid[1][tu]
        if ind == tot_ind:
            print('Agent trapped!')
            is_trapped = True
            self.trapped += 1

        if not collision and not is_trapped:
            self.actions.append(action)
            self.last_action_dir = (action_dest[0] - x, action_dest[1] - y)
            self.state.update({action_dest: self.seq[stone_idx]})
            # update H or P plain
            self.grid[p_to_int[self.seq[stone_idx]]][action_dest] = 1
            # update primary connect plain
            connect_idx = (int((x + action_dest[0]) / 2), int((y + action_dest[1]) / 2))
            self.grid[2][connect_idx] = 1
            # update H-H connect plain
            if self.seq[stone_idx] == 'H':
                adj_coord = self._get_adjacent_coords(action_dest)
                for co in adj_coord:
                    if self.grid[0][co] == 0:
                        continue
                    connect = (int((action_dest[0] + co[0]) / 2), int((action_dest[1] + co[1]) / 2))
                    if self.grid[2][connect] == 1:
                        continue
                    self.grid[3][connect] = 1

        done = (len(self.state) == len(self.seq) or is_trapped)
        reward = self._compute_reward(is_trapped, collision, done)
        info = {
            'chain_length': len(self.state),
            'seq_length': len(self.seq),
            'collisions': self.collisions,
            'actions': [ACTION_TO_STR[i] for i in self.actions],
            'is_trapped': is_trapped,
            'state_chain': self.state
        }

        return self.grid, reward, done, info

    def reset(self):
        """Resets the environment"""
        self.state = OrderedDict({self.midpoint: self.seq[0]})
        self.actions = []
        self.collisions = 0
        self.trapped = 0
        self.grid = np.zeros(shape=(4, self.grid_length, self.grid_length), dtype=int)
        # Automatically assign first element into grid
        self.grid[p_to_int[self.seq[0]]][self.midpoint] = 1

        return self.grid

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        desc = self.grid.astype(str)

        # Convert everything to human-readable symbols
        desc[desc == '0'] = '*'
        desc[0][desc[0] == '1'] = 'H'
        desc[1][desc[1] == '1'] = 'P'

        # Obtain all x-y indices of elements
        x_free, y_free = np.where(np.logical_or(desc[0] == '*', desc[1] == '*'))
        x_h, y_h = np.where(desc[0] == 'H')
        x_p, y_p = np.where(desc[1] == 'P')
        x_c, y_c = np.where(desc[3] == '1')

        # Decode if possible
        desc.tolist()
        try:
            desc = [[c.decode('utf-8') for c in line] for line in desc]
        except AttributeError:
            pass

        desc = desc[0]
        # All unfilled spaces are gray
        for unfilled_coords in zip(x_free, y_free):
            desc[unfilled_coords] = utils.colorize(desc[unfilled_coords], "gray")

        # All hydrophobic molecules are bold-green
        for hmol_coords in zip(x_h, y_h):
            desc[hmol_coords] = utils.colorize(desc[hmol_coords], "green")
        desc[self.midpoint] = utils.colorize(desc[self.midpoint], "yellow")

        # All polar molecules are cyan
        for pmol_coords in zip(x_p, y_p):
            desc[pmol_coords] = 'P'
            desc[pmol_coords] = utils.colorize(desc[pmol_coords], "cyan")

        for con_coords in zip(x_c, y_c):
            desc[con_coords] = '='
            desc[con_coords] = utils.colorize(desc[con_coords], "red")

        # Provide prompt for last action
        if self.last_action is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Up", "Right"][self.last_action]))
        else:
            outfile.write("\n")

        # Draw desc
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            return outfile

    def _get_possible_dest(self, coords):
        x, y = coords
        forward = (x + self.last_action_dir[0], y + self.last_action_dir[1])
        left = (x + left_dir[self.last_action_dir][0], y + left_dir[self.last_action_dir][1])
        right = (x - left_dir[self.last_action_dir][0], y - left_dir[self.last_action_dir][1])

        return forward, left, right

    def _get_adjacent_coords(self, coords):
        x, y = coords
        adjacent_coords = []
        if x - 2 >= 0:
            adjacent_coords.append((x - 2, y))
        if x + 2 < self.grid_length:
            adjacent_coords.append((x + 2, y))
        if y - 2 >= 0:
            adjacent_coords.append((x, y - 2))
        if y + 2 < self.grid_length:
            adjacent_coords.append((x, y + 2))

        return adjacent_coords

    def _compute_reward(self, is_trapped, collision, done):
        state_reward = np.sum(self.grid[3]) if done else 0
        collision_penalty = self.collision_penalty if collision else 0
        actual_trap_penalty = -floor(len(self.seq) * self.trap_penalty) if is_trapped else 0

        return state_reward + collision_penalty + actual_trap_penalty

    def imagine_next(self, action, state, char):
        # the action destination of this move
        h_pos = state[0] - state[4]
        p_pos = state[1] - state[5]
        if np.sum(h_pos) == 1:
            x, y = np.where(h_pos == 1)
            x, y = x[0], y[0]
        else: # np.sum(p_pos) == 1:
            x, y = np.where(p_pos == 1)
            x, y = x[0], y[0]
        # the action destination of last move
        h_prevpos = state[4] - state[8]
        p_prevpos = state[5] - state[9]
        if np.sum(h_prevpos) == 1:
            x_prev, y_prev = np.where(h_prevpos == 1)
            x_prev, y_prev = x_prev[0], y_prev[0]
            last_dir = (x - x_prev, y - y_prev)
        elif np.sum(p_prevpos) == 1:
            x_prev, y_prev = np.where(p_prevpos == 1)
            x_prev, y_prev = x_prev[0], y_prev[0]
            last_dir = (x - x_prev, y - y_prev)
        else:
            last_dir = (-2, 0)

        forward = (x + last_dir[0], y + last_dir[1])
        left = (x + left_dir[last_dir][0], y + left_dir[last_dir][1])
        right = (x - left_dir[last_dir][0], y - left_dir[last_dir][1])
        possible_dest = forward, left, right
        action_dest = possible_dest[action]
        if not (0 <= action_dest[0] <= self.grid_length - 1 and 0 <= action_dest[1] <= self.grid_length - 1):
            return None

        # determine whether collide
        if state[0][action_dest] == 1 or state[1][action_dest] == 1:
            return None

        # determine whether trapped
        # ind = 0
        # tot_ind = 0
        # for tu in possible_dest:
        #     if not (0 <= tu[0] <= self.grid_length - 1 and 0 <= tu[1] <= self.grid_length - 1):
        #         continue
        #     tot_ind += 1
        #     ind += state[0][tu] + state[1][tu]
        # if ind == tot_ind:
        #     return None

        # update H or P plain
        grid = np.copy(state)
        grid[p_to_int[char]][action_dest] = 1
        # update primary connect plain
        connect_idx = (int((x + action_dest[0]) / 2), int((y + action_dest[1]) / 2))
        grid[2][connect_idx] = 1
        # update H-H connect plain
        if char == 'H':
            adj_coord = self._get_adjacent_coords(action_dest)
            for co in adj_coord:
                if grid[0][co] == 0:
                    continue
                connect = (int((action_dest[0] + co[0]) / 2), int((action_dest[1] + co[1]) / 2))
                if grid[2][connect] == 1:
                    continue
                grid[3][connect] = 1

        return grid[0:4]
