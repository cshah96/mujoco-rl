"""
Task 3: Hopper-v5 -- a one-legged robot that must learn to hop forward.

    python scripts/train_hopper.py                       # 1M steps, ~1-2 h on CPU
    python scripts/watch.py --model models/Hopper-v5_sac.zip

Observation (11): torso height, torso angle, thigh/leg/foot joint angles (5 positions)
                  + torso x/z/angle velocities and the 3 joint velocities (6),
                  velocities clipped to +-10. Torso x position is excluded so
                  the policy behaves the same anywhere along the track.
Action       (3): torques for the thigh, leg and foot joints, each in [-1, 1]
Reward          : x_velocity            (move forward)
                  + 1                   (healthy bonus while upright)
                  - 1e-3 * sum(a^2)     (energy / control cost)
Episode end     : torso height < 0.7 m, |torso angle| > 0.2 rad, any state
                  value > 100 (fell over / blew up), or 1000 steps
Good return     : ~3000+ (hops steadily to the end of the episode)

The first real locomotion task: the agent has to discover a gait purely from
reward. We switch to SAC here -- it reuses past experience from a replay buffer
and is far more sample-efficient than PPO on locomotion.
"""

import argparse

import common  # noqa: F401  (must be first: segfault fix + headless rendering)


def main():
    p = common.common_args(argparse.ArgumentParser(description=__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter),
                           default_timesteps=1_000_000, default_n_envs=1)
    args = p.parse_args()

    common.train(
        "Hopper-v5", "sac", args.timesteps,
        n_envs=args.n_envs, seed=args.seed, run_name=args.run_name,
        checkpoint_every=args.checkpoint_every or 100_000, device=args.device,
        video_after=not args.no_video,
        # SAC learns its own reward scale via the entropy temperature, so we
        # only normalize observations, not rewards.
        norm_reward=False,
        algo_kwargs=dict(
            learning_starts=10_000,  # act randomly first to fill the replay buffer
            buffer_size=1_000_000,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,               # soft target-network update rate
            ent_coef="auto",         # automatically tuned exploration bonus
        ),
    )


if __name__ == "__main__":
    main()
