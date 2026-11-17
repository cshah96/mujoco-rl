"""
Evaluate every checkpoint of a run and promote the best one to be the run's
main policy (models/<run>.zip), which is what watch.py and the README use.

    python scripts/pick_best.py --run Hopper-v5_sac
    python scripts/pick_best.py --run AntConstantSpeed-v0_sac --episodes 10

Why: off-policy algorithms like SAC don't improve monotonically -- performance
often oscillates late in training, so the *last* policy is not always the
best one. (Here, Hopper's 900k-step checkpoint completed 10/10 episodes while
the final 1M-step policy completed only 3/10.)

The policy that was originally saved at the end of training is kept as
models/<run>_final.zip (+ its _vecnormalize.pkl) so nothing is lost.
Scoring uses the deterministic policy, averaged over --episodes episodes.
"""

import argparse
import glob
import os
import re
import shutil

import common  # noqa: F401  (must be first: segfault fix + headless rendering)
import numpy as np


def evaluate(model_path, episodes):
    env, model, _ = common.load_policy(model_path)
    returns, lengths, speeds = [], [], []
    for _ in range(episodes):
        obs = env.reset()
        done, ret, n, v = False, 0.0, 0, []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            ret += float(reward[0])
            n += 1
            if "x_velocity" in infos[0]:
                v.append(infos[0]["x_velocity"])
        returns.append(ret)
        lengths.append(n)
        speeds.append(np.mean(v) if v else np.nan)
    env.close()
    return np.mean(returns), np.mean(lengths), np.nanmean(speeds) if any(
        not np.isnan(s) for s in speeds) else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Run name, e.g. Hopper-v5_sac")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--dry-run", action="store_true", help="Only print the scores")
    args = p.parse_args()

    paths = common.run_paths(args.run)
    final_model = paths["model"]
    kept_final = final_model.replace(".zip", "_final.zip")
    # Score the end-of-training policy (wherever it currently lives) + all checkpoints.
    candidates = [kept_final if os.path.exists(kept_final) else final_model]
    ckpts = glob.glob(os.path.join(paths["checkpoints"], f"{args.run}_*_steps.zip"))
    candidates += sorted(ckpts, key=lambda c: int(re.search(r"_(\d+)_steps", c).group(1)))

    results = []
    final_score = None
    for c in candidates:
        if not os.path.exists(c):
            continue
        ret, length, speed = evaluate(c, args.episodes)
        results.append((ret, c))
        if final_score is None:
            final_score = ret  # candidates[0] is the end-of-training policy
        extra = f"  speed {speed:5.2f} m/s" if speed is not None else ""
        print(f"{os.path.relpath(c, common.REPO_ROOT):70s} return {ret:8.1f}  "
              f"len {length:6.1f}{extra}", flush=True)

    best_ret, best = max(results)
    print(f"\nBest: {os.path.relpath(best, common.REPO_ROOT)} (mean return {best_ret:.1f})")
    # The last checkpoint holds the same weights as the final policy; only
    # promote when another checkpoint scores clearly (>1%) better.
    if args.dry_run or best_ret <= final_score + 0.01 * abs(final_score):
        print("Main policy unchanged.")
        return

    # Keep the original end-of-training policy, then copy the best checkpoint over.
    if not os.path.exists(kept_final):
        shutil.copy2(final_model, kept_final)
        shutil.copy2(paths["vecnormalize"], kept_final.replace(".zip", "_vecnormalize.pkl"))
    shutil.copy2(best, final_model)
    shutil.copy2(common._stats_path_for(best), paths["vecnormalize"])
    print(f"Promoted it to {os.path.relpath(final_model, common.REPO_ROOT)}; "
          f"original end-of-training policy kept as {os.path.relpath(kept_final, common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
