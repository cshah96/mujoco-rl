"""
Play back a trained policy -- live in a MuJoCo window, or rendered to .mp4.

    # live window (needs a display)
    python scripts/watch.py --model models/Hopper-v5_sac.zip

    # headless: render to a video file (works over SSH / on a server)
    python scripts/watch.py --model models/Hopper-v5_sac.zip --mode video

    # an intermediate checkpoint saved during training
    python scripts/watch.py --model models/Hopper-v5_sac_checkpoints/Hopper-v5_sac_300000_steps.zip

The env id, env settings and algorithm are read from the <run>_config.json
saved next to the model, and the observation-normalization stats from
<run>_vecnormalize.pkl, so the policy sees exactly what it saw in training.
"""

import argparse

import common  # noqa: F401  (must be first: segfault fix + headless rendering)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Path to a saved .zip policy")
    p.add_argument("--mode", default="human", choices=["human", "video"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--video-path", default=None,
                   help="Default: videos/<model name>.mp4")
    p.add_argument("--fps", type=int, default=None, help="Default: real time")
    p.add_argument("--env", default=None,
                   help="Only needed for models without a _config.json")
    p.add_argument("--algo", default=None, choices=list(common.ALGOS.keys()),
                   help="Only needed if it can't be read from config/filename")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions (with exploration noise) instead of the mean")
    args = p.parse_args()

    common.play(args.model, mode=args.mode, episodes=args.episodes,
                video_path=args.video_path, fps=args.fps, env_id=args.env,
                algo=args.algo, deterministic=not args.stochastic)


if __name__ == "__main__":
    main()
