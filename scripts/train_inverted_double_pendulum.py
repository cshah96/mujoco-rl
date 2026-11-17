"""
Task 2: InvertedDoublePendulum-v5 -- balance TWO hinged poles on a cart.

    python scripts/train_inverted_double_pendulum.py     # 200k steps, ~25 min on CPU
    python scripts/watch.py --model models/InvertedDoublePendulum-v5_sac.zip

Observation (9): cart x, sin/cos of both pole angles, cart + pole velocities
                 (clipped to +-10), and the constraint force on the cart
Action      (1): force on the cart, in [-1, 1]
Reward         : +10 alive bonus each step
                 - (0.01 * tip_x^2 + (tip_y - 2)^2)   keep the tip high & centred
                 - (1e-3 * w1^2 + 5e-3 * w2^2)        don't swing the poles fast
Episode end    : tip of the top pole drops to height <= 1 m, or 1000 steps
Max return     : ~9360

Much less stable than the single pendulum: the system is chaotic, so small
mistakes amplify quickly. We use SAC here: in this repo, PPO (1M steps, tuned
zoo hyperparameters) only balanced from ~75-90% of start states, while SAC
with defaults balanced 20/20 test episodes at the max return after 200k steps.
SAC's replay buffer lets it keep re-learning from the rare, hard starting
states instead of discarding them after one update.
"""

import argparse

import common  # noqa: F401  (must be first: segfault fix + headless rendering)


def main():
    p = common.common_args(argparse.ArgumentParser(description=__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter),
                           default_timesteps=200_000, default_n_envs=1)
    args = p.parse_args()

    common.train(
        "InvertedDoublePendulum-v5", "sac", args.timesteps,
        n_envs=args.n_envs, seed=args.seed, run_name=args.run_name,
        checkpoint_every=args.checkpoint_every or 50_000, device=args.device,
        video_after=not args.no_video,
        norm_reward=False,
        # SB3 SAC defaults: buffer 1M, batch 256, lr 3e-4, gamma 0.99,
        # tau 0.005, auto entropy; we only lengthen the random warm-up.
        algo_kwargs=dict(learning_starts=10_000),
    )


if __name__ == "__main__":
    main()
