"""
Generic trainer for any Gymnasium MuJoCo env, with algorithm defaults.
The train_<task>.py scripts are the tuned, documented versions; use this one
to experiment with envs that don't have a dedicated script yet.

    python scripts/train.py --env Walker2d-v5 --algo sac --timesteps 1000000
    python scripts/train.py --env InvertedPendulum-v5 --algo ppo --n-envs 4
"""

import argparse

import common  # noqa: F401  (must be first: segfault fix + headless rendering)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="InvertedPendulum-v5", help="Gymnasium env id")
    p.add_argument("--algo", default="ppo", choices=common.ALGOS.keys())
    common.common_args(p, default_timesteps=100_000, default_n_envs=4)
    args = p.parse_args()

    common.train(
        args.env, args.algo, args.timesteps,
        n_envs=args.n_envs, seed=args.seed, run_name=args.run_name,
        checkpoint_every=args.checkpoint_every, device=args.device,
        video_after=not args.no_video,
        norm_reward=(args.algo == "ppo"),
    )


if __name__ == "__main__":
    main()
