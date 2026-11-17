"""
Shared setup + helpers for every script in this repo.

IMPORT THIS MODULE FIRST, before gymnasium / mujoco / torch / stable_baselines3:

    import common  # noqa: F401  (must be the first import)

It does two things at import time that have to happen before those libraries
load:

1. The segfault fix.
   The shared venv has TensorFlow installed (it is not one of our
   requirements). Stable-Baselines3's TensorBoard logger imports `tensorboard`,
   which opportunistically imports TensorFlow if it can find it. Later, the
   first torch optimizer lazily loads Triton's native library, and Triton's
   native code segfaults when TensorFlow's native code is already loaded in the
   same process (the reverse order is fine). We never use TensorFlow, so we
   simply make it un-importable for this process. TensorBoard then falls back
   to its built-in, TF-free writer, and logging works exactly the same (and
   startup is several seconds faster).

2. Headless rendering.
   If there's no display (SSH, a server, a background job), MuJoCo can't open
   an OpenGL window. We switch it to EGL, which renders on the GPU with no
   window, so recording videos works anywhere.
"""

import os
import sys

# --- 1. segfault fix: keep TensorFlow out of this process (see docstring) ----
sys.modules["tensorflow"] = None  # `import tensorflow` now raises ImportError
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# --- 2. headless rendering ---------------------------------------------------
if not os.environ.get("DISPLAY") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

# Registers our custom env ids (e.g. AntConstantSpeed-v0) with gymnasium.
import ant_constant_speed  # noqa: E402,F401

ALGOS = {"ppo": PPO, "sac": SAC}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
VIDEOS_DIR = os.path.join(REPO_ROOT, "videos")


def run_paths(run_name, models_dir=MODELS_DIR):
    """All files belonging to one trained policy share the same prefix."""
    prefix = os.path.join(models_dir, run_name)
    return {
        "model": prefix + ".zip",                    # the policy network weights
        "vecnormalize": prefix + "_vecnormalize.pkl",  # obs/reward normalization stats
        "config": prefix + "_config.json",           # env id/kwargs + algo, for playback
        "checkpoints": prefix + "_checkpoints",      # periodic snapshots during training
    }


def train(
    env_id,
    algo,
    timesteps,
    *,
    env_kwargs=None,
    algo_kwargs=None,
    n_envs=1,
    norm_obs=True,
    norm_reward=True,
    seed=0,
    run_name=None,
    checkpoint_every=0,
    device="cpu",
    video_after=True,
):
    """Train a policy headlessly, save everything needed to replay it later.

    Nothing is rendered during training -- the simulator just runs as fast as
    the CPU allows. When training finishes we (optionally) record a short
    video of the final policy so you can check the result without a display.
    """
    env_kwargs = env_kwargs or {}
    algo_kwargs = algo_kwargs or {}
    run_name = run_name or f"{env_id}_{algo}"
    paths = run_paths(run_name)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # n_envs copies of the simulator, stepped together. Each `env.step` on the
    # vector env returns n_envs transitions, so data collection is faster.
    vec_env = make_vec_env(env_id, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs)

    # Running mean/std normalization. MuJoCo observations mix very different
    # scales (angles ~1, velocities ~10, contact forces ~100s); networks learn
    # much better when every input is roughly zero-mean, unit-variance.
    vec_env = VecNormalize(vec_env, norm_obs=norm_obs, norm_reward=norm_reward)

    model = ALGOS[algo](
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        tensorboard_log=LOGS_DIR,
        # Small MLPs: per-step GPU launch overhead outweighs the compute
        # benefit, so CPU is usually faster. Pass --device cuda to compare.
        device=device,
        **algo_kwargs,
    )

    callbacks = []
    if checkpoint_every:
        # save_freq counts calls to env.step on the *vector* env, hence / n_envs.
        callbacks.append(CheckpointCallback(
            save_freq=max(checkpoint_every // n_envs, 1),
            save_path=paths["checkpoints"],
            name_prefix=run_name,
            save_vecnormalize=True,
        ))

    # Written up front so checkpoints can be played back while training runs.
    with open(paths["config"], "w") as f:
        json.dump({"env_id": env_id, "env_kwargs": env_kwargs, "algo": algo,
                   "timesteps": timesteps, "seed": seed}, f, indent=2)

    model.learn(total_timesteps=timesteps, tb_log_name=run_name,
                callback=callbacks, progress_bar=True)

    model.save(paths["model"])
    vec_env.save(paths["vecnormalize"])
    vec_env.close()

    print(f"\nSaved policy        -> {paths['model']}")
    print(f"Saved normalization -> {paths['vecnormalize']}")
    print(f"Saved run config    -> {paths['config']}")

    if video_after:
        video_path = os.path.join(VIDEOS_DIR, f"{run_name}.mp4")
        play(paths["model"], mode="video", episodes=1, video_path=video_path)

    rel = os.path.relpath(paths["model"], REPO_ROOT)
    print(f"\nReplay live:      python scripts/watch.py --model {rel}")
    print(f"Replay to video:  python scripts/watch.py --model {rel} --mode video")
    print(f"Training curves:  tensorboard --logdir logs")
    return paths


def load_policy(model_path, env_id=None, algo=None, env_kwargs=None, render_mode=None):
    """Rebuild (env, model) exactly as they were at the end of training."""
    config_path = model_path.replace(".zip", "_config.json")
    if model_path.endswith(".zip") and "_checkpoints" in model_path:
        # checkpoint files: <run>_checkpoints/<run>_<steps>_steps.zip
        ckpt_dir = os.path.dirname(model_path)
        config_path = ckpt_dir.replace("_checkpoints", "_config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    env_id = env_id or config.get("env_id")
    algo = algo or config.get("algo") or _guess_algo(model_path)
    env_kwargs = env_kwargs if env_kwargs is not None else config.get("env_kwargs", {})
    if env_id is None:
        raise ValueError(f"No config at {config_path}; pass --env explicitly")

    env = DummyVecEnv([lambda: gym.make(env_id, render_mode=render_mode, **env_kwargs)])

    # The policy was trained on *normalized* observations, so it must see the
    # same normalization at playback (forgetting this is a classic RL bug).
    stats_path = _stats_path_for(model_path)
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False     # freeze the running statistics
        env.norm_reward = False  # report true, un-normalized rewards
    else:
        print(f"Warning: no normalization stats at {stats_path}; "
              f"policy will see raw observations and may misbehave.")

    model = ALGOS[algo].load(model_path, device="cpu")
    return env, model, env_id


def play(model_path, mode="human", episodes=3, video_path=None, fps=None,
         env_id=None, algo=None, deterministic=True):
    """Run a trained policy: live window (mode="human") or to an .mp4 (mode="video")."""
    render_mode = "human" if mode == "human" else "rgb_array"
    env, model, env_id = load_policy(model_path, env_id=env_id, algo=algo,
                                     render_mode=render_mode)
    raw_env = env.envs[0] if hasattr(env, "envs") else env.venv.envs[0]
    if fps is None:
        # Real-time playback: one rendered frame per env step.
        fps = int(round(1.0 / raw_env.unwrapped.dt))

    frames = []
    for ep in range(episodes):
        obs = env.reset()
        done, ep_reward, steps, speeds = False, 0.0, 0, []
        while not done:
            # deterministic=True -> use the mean action, no exploration noise.
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            ep_reward += float(reward[0])
            steps += 1
            if "x_velocity" in infos[0]:
                speeds.append(infos[0]["x_velocity"])
            if mode == "video":
                frames.append(raw_env.render())
        msg = f"Episode {ep + 1}: return = {ep_reward:8.2f}   length = {steps}"
        if speeds:
            msg += f"   mean forward speed = {np.mean(speeds):.2f} m/s"
        print(msg)

    env.close()
    if mode == "video":
        video_path = video_path or os.path.join(
            VIDEOS_DIR, os.path.basename(model_path).replace(".zip", ".mp4"))
        os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
        imageio.mimsave(video_path, frames, fps=fps)
        print(f"Saved video -> {video_path}")


def _stats_path_for(model_path):
    if "_checkpoints" in model_path:
        # CheckpointCallback names: <run>_<N>_steps.zip / <run>_vecnormalize_<N>_steps.pkl
        d, name = os.path.split(model_path)
        run = os.path.basename(d).replace("_checkpoints", "")
        steps = name.replace(run + "_", "").replace(".zip", "")
        return os.path.join(d, f"{run}_vecnormalize_{steps}.pkl")
    return model_path.replace(".zip", "_vecnormalize.pkl")


def _guess_algo(model_path):
    for name in ALGOS:
        if f"_{name}" in os.path.basename(model_path):
            return name
    raise ValueError("Could not guess algo from model filename; pass --algo")


def common_args(parser, default_timesteps, default_n_envs=1):
    """CLI flags shared by every train_*.py script."""
    parser.add_argument("--timesteps", type=int, default=default_timesteps,
                        help="Total environment steps to train for")
    parser.add_argument("--n-envs", type=int, default=default_n_envs,
                        help="Parallel simulator copies used to collect experience")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None,
                        help="Name prefix for saved files (default: <env>_<algo>)")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Also save a snapshot every N env steps (0 = off)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip recording a video of the final policy")
    return parser
