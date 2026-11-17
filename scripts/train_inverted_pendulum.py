"""
Task 1: InvertedPendulum-v5 -- balance a pole on a sliding cart.

    python scripts/train_inverted_pendulum.py            # ~1-2 min on CPU
    python scripts/watch.py --model models/InvertedPendulum-v5_ppo.zip

Observation (4): cart position, pole angle, cart velocity, pole angular velocity
Action      (1): force on the cart, in [-3, 3]
Reward         : +1 for every step the pole is within 0.2 rad (~11 deg) of upright
Episode end    : pole tips past 0.2 rad (terminated) or 1000 steps (truncated)
Max return     : 1000

PPO with stable-baselines3 defaults solves this easily -- it is the "hello
world" of continuous control, a sanity check that the whole pipeline works.
"""

import argparse

import common  # noqa: F401  (must be first: segfault fix + headless rendering)


def main():
    p = common.common_args(argparse.ArgumentParser(description=__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter),
                           default_timesteps=100_000, default_n_envs=4)
    args = p.parse_args()

    common.train(
        "InvertedPendulum-v5", "ppo", args.timesteps,
        n_envs=args.n_envs, seed=args.seed, run_name=args.run_name,
        checkpoint_every=args.checkpoint_every, device=args.device,
        video_after=not args.no_video,
        # PPO defaults: n_steps=2048 per env, batch_size=64, n_epochs=10,
        # lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2.
        algo_kwargs={},
    )


if __name__ == "__main__":
    main()
