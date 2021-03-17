"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import retro
from gym.spaces import Box
from gym import Wrapper
import cv2
import numpy as np
import subprocess as sp
import torch.multiprocessing as mp

STATES = {
    "1-1": "GreenHillZone.Act1",
    "1-2": "GreenHillZone.Act2",
    "1-3": "GreenHillZone.Act3",
    "2-1": "MarbleZone.Act1",
    "2-2": "MarbleZone.Act2",
    "2-3": "MarbleZone.Act3",
    "3-1": "SpringYardZone.Act1",
    "3-2": "SpringYardZone.Act2",
    "3-3": "SpringYardZone.Act3",
    "4-1": "LabyrinthZone.Act1",
    "4-2": "LabyrinthZone.Act2",
    "4-3": "LabyrinthZone.Act3",
    "5-1": "StarLightZone.Act1",
    "5-2": "StarLightZone.Act2",
    "5-3": "StarLightZone.Act3",
    "6-1": "ScrapBrainZone.Act1",
    "6-2": "ScrapBrainZone.Act2",
    "6-3": "ScrapBrainZone.Act3",
}
# Work for 1-1, 1-2
ACTION_MAPPING = {
    # Left
    0: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    # Right
    1: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    # Right, Down
    2: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    # Right, B
    3: [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
}

# ACTION_MAPPING = {
#             # Do nothing
#             0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             # Left
#             1: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#             # Right
#             2: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#             # Left, Down
#             3: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
#             # Right, Down
#             4: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
#             # Left, B
#             5: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#             # Right, B
#             6: [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#             # Down
#             7: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#             # Down, B
#             8: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#             # B
#             9: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#         }

class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "60", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env=None, act=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.env.reset()
        _, _, _, info = self.env.step(env.action_space.sample())
        self.init_pos = info["x"]
        self.init_score = info["score"]
        self.init_rings = info["rings"]
        self.init_lives = info["lives"]
        self.env.reset()
        self.curr_pos = self.init_pos
        self.curr_score = self.init_score
        self.curr_rings = self.init_rings
        self.curr_lives = self.init_lives
        self.act = act

        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state, _, done, info = self.env.step(action)
        if self.monitor:
            self.monitor.record(state)
        state = process_frame(state)
        reward = min(max((info['x'] - self.curr_pos), 0), 2)
        self.curr_pos = max(info['x'], self.curr_pos)
        reward += (info["score"] - self.curr_score) / 40
        self.curr_score = info["score"]
        # reward += min(max(info["rings"] - self.curr_rings, -2), 2)
        # self.curr_rings = info["rings"]
        reward -= 1 / 600
        if info["lives"] < 3:
            return state, -5, True, info
        if info["act"] == self.act:
            return state, 5, True, info
        return state, reward / 10, done, info

    def reset(self):
        self.curr_pos = self.init_pos
        self.curr_score = self.init_score
        self.curr_rings = self.init_rings
        self.curr_lives = self.init_lives
        return process_frame(self.env.reset())


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_states = []
        for i in range(self.skip):
            state, reward, done, info = self.env.step(ACTION_MAPPING[action])
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if done:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        max_state = np.max(np.concatenate(last_states, 0), 0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states[None, :, :, :].astype(np.float32)


def create_train_env(zone, act, output_path=None):
    env = retro.make('SonicTheHedgehog-Genesis', STATES["{}-{}".format(zone, act)], use_restricted_actions=retro.Actions.FILTERED)
    if output_path:
        monitor = Monitor(320, 224, output_path)
    else:
        monitor = None

    env = CustomReward(env, act, monitor)
    env = CustomSkipFrame(env)
    return env


class MultipleEnvironments:
    def __init__(self, zone, act, num_envs, output_path=None):
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        env = create_train_env(zone, act, output_path=output_path)
        self.num_states = env.observation_space.shape[0]
        env.close()
        self.num_actions = len(ACTION_MAPPING)
        for index in range(num_envs):
            process = mp.Process(target=self.run, args=(index, zone, act, output_path))
            process.start()
            self.env_conns[index].close()

    def run(self, index, zone, act, output_path):
        env = create_train_env(zone, act, output_path=output_path)
        self.agent_conns[index].close()
        while True:
            request, action = self.env_conns[index].recv()
            if request == "step":
                self.env_conns[index].send(env.step(action.item()))
            elif request == "reset":
                self.env_conns[index].send(env.reset())
            else:
                raise NotImplementedError
