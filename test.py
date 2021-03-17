"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
import torch.nn.functional as F
from src.env import create_train_env, ACTION_MAPPING, STATES
from src.model import PPO


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Contra Nes""")
    parser.add_argument("--zone", type=int, default=1)
    parser.add_argument("--act", type=int, default=2)
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default="output")
    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    env = create_train_env(opt.zone, opt.act,
                           output_path="{}/video_{}.mp4".format(opt.output_path,
                                                                STATES["{}-{}".format(opt.zone, opt.act)]))
    model = PPO(env.observation_space.shape[0], len(ACTION_MAPPING))
    if torch.cuda.is_available():
        model.load_state_dict(
            torch.load("{}/PPO_SonicTheHedgehog_{}".format(opt.saved_path, STATES["{}-{}".format(opt.zone, opt.act)])))
        model.cuda()
    else:
        model.load_state_dict(
            torch.load("{}/PPO_SonicTheHedgehog_{}".format(opt.saved_path, STATES["{}-{}".format(opt.zone, opt.act)]),
                       map_location=lambda storage, loc: storage))
    model.eval()
    state = torch.from_numpy(env.reset())
    while True:
        if torch.cuda.is_available():
            state = state.cuda()
        logits, value = model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        env.render()
        if done and info["act"] == opt.act:
            print("Map {} is completed".format(STATES["{}-{}".format(opt.zone, opt.act)]))
            break


if __name__ == "__main__":
    opt = get_args()
    test(opt)
