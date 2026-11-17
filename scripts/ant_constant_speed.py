"""
AntConstantSpeed-v0: the standard Ant-v5 robot, but rewarded for walking
forward at a *target* speed instead of as fast as possible.

Why a custom reward is needed:
  Ant-v5's reward includes `forward_reward = x_velocity`, which grows without
  bound -- the faster the ant goes the more reward it gets, so trained ants
  sprint/lunge as fast as physics allows. To get a steady walk at, say,
  1 m/s, we replace that term with a *tracking* reward that peaks at the
  target speed and falls off on either side:

      tracking = exp( -(v_x - v_target)^2 / sigma^2 )      in [0, 1]

  This is the same velocity-tracking kernel used by most modern legged-robot
  RL work (e.g. ETH's legged_gym / ANYmal locomotion). We also penalize
  sideways velocity so it walks straight along +x instead of drifting.

Full per-step reward:

      reward = w_track * tracking                 # walk at the target speed
             - w_lateral * v_y^2                  # don't drift sideways
             + healthy_reward                     # +1 for staying upright (from Ant-v5)
             - ctrl_cost                          # 0.5 * sum(action^2)   (from Ant-v5)
             - contact_cost                       # 5e-4 * sum(F_ext^2)   (from Ant-v5)

The observation/action spaces and termination rule are unchanged from Ant-v5.
"""

import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.ant_v5 import AntEnv


class AntConstantSpeedEnv(AntEnv):
    def __init__(
        self,
        target_speed: float = 1.0,
        tracking_weight: float = 2.0,
        tracking_sigma: float = 0.5,
        lateral_weight: float = 0.5,
        **kwargs,
    ):
        # Ant-v5 forward reward turned off; everything else stays as default.
        kwargs.setdefault("forward_reward_weight", 0.0)
        super().__init__(**kwargs)
        self.target_speed = target_speed
        self.tracking_weight = tracking_weight
        self.tracking_sigma = tracking_sigma
        self.lateral_weight = lateral_weight

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # `reward` from Ant-v5 here is healthy - ctrl - contact (forward weight is 0).
        vx, vy = info["x_velocity"], info["y_velocity"]
        tracking = np.exp(-((vx - self.target_speed) / self.tracking_sigma) ** 2)
        tracking_reward = self.tracking_weight * tracking
        lateral_cost = self.lateral_weight * vy ** 2
        reward = reward + tracking_reward - lateral_cost

        info["reward_tracking"] = tracking_reward
        info["reward_lateral"] = -lateral_cost
        info["speed_error"] = vx - self.target_speed
        return obs, reward, terminated, truncated, info


if "AntConstantSpeed-v0" not in gym.registry:
    gym.register(
        id="AntConstantSpeed-v0",
        entry_point=AntConstantSpeedEnv,
        max_episode_steps=1000,  # same 1000-step (50 s) episodes as Ant-v5
    )
