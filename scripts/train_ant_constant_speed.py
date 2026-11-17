"""
Task 4: AntConstantSpeed-v0 -- a 4-legged 3D ant that walks forward at a
steady target speed (default 1.0 m/s) instead of sprinting.

    python scripts/train_ant_constant_speed.py                    # 1M steps, ~2-3 h on CPU
    python scripts/train_ant_constant_speed.py --target-speed 0.5 # slower walk
    python scripts/watch.py --model models/AntConstantSpeed-v0_sac.zip

Observation (105): torso height + orientation quaternion + 8 joint angles (13)
                   + torso linear/angular velocity + 8 joint velocities (14)
                   + external contact forces on 13 bodies x 6 (78)
Action        (8): torques for the hip and ankle of each of the 4 legs, in [-1, 1]
Reward           : 2 * exp(-(v_x - v_target)^2 / 0.5^2)   (track the target speed)
                   - 0.5 * v_y^2                          (walk straight)
                   + 1                                    (healthy bonus)
                   - 0.5 * sum(a^2)                       (control cost)
                   - 5e-4 * sum(F_contact^2)              (contact cost)
Episode end      : torso height outside [0.2, 1.0] m (flipped/jumped), or 1000 steps
Good result      : mean forward speed within ~0.1 m/s of the target, return ~2500+

See scripts/ant_constant_speed.py for the env definition.
"""

import argparse

import common  # noqa: F401  (must be first: segfault fix + headless rendering)


def main():
    p = common.common_args(argparse.ArgumentParser(description=__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter),
                           default_timesteps=1_000_000, default_n_envs=1)
    p.add_argument("--target-speed", type=float, default=1.0,
                   help="Desired forward speed in m/s")
    args = p.parse_args()

    run_name = args.run_name or f"AntConstantSpeed-v0_sac"
    if args.run_name is None and args.target_speed != 1.0:
        run_name = f"AntConstantSpeed-v0_{args.target_speed:g}ms_sac"

    common.train(
        "AntConstantSpeed-v0", "sac", args.timesteps,
        env_kwargs={"target_speed": args.target_speed},
        n_envs=args.n_envs, seed=args.seed, run_name=run_name,
        checkpoint_every=args.checkpoint_every or 100_000, device=args.device,
        video_after=not args.no_video,
        norm_reward=False,
        algo_kwargs=dict(
            learning_starts=10_000,
            buffer_size=1_000_000,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
        ),
    )


if __name__ == "__main__":
    main()
