# MuJoCo RL Learning Bed

A hands-on sandbox for learning reinforcement learning (RL) on MuJoCo
physics simulations. It has one ready-to-run, documented training script per
task, ordered from "balance a stick" to "walk on four legs at a set speed":

| # | Task | Script | Algorithm | Wall time (CPU) |
|---|---|---|---|---|
| 1 | Inverted pendulum | `scripts/train_inverted_pendulum.py` | PPO | ~1 min |
| 2 | Inverted double pendulum | `scripts/train_inverted_double_pendulum.py` | SAC | ~25 min |
| 3 | Hopper (one-legged hopping) | `scripts/train_hopper.py` | SAC | ~1–2 h |
| 4 | Ant walking at a constant speed | `scripts/train_ant_constant_speed.py` | SAC | ~2–3 h |

Every script trains **headless**: nothing is drawn on screen, so it runs at
full speed over SSH, in `tmux`, or as a background job. Every script saves
everything needed to **play the policy back later**, live or as a video
(see [Playing back a trained policy](#playing-back-a-trained-policy)).

---

## Contents

1. [Quick start](#quick-start)
2. [Project layout](#project-layout)
3. [The segfault fix (read this before writing new scripts)](#the-segfault-fix)
4. [RL in one page: the concepts these scripts use](#rl-in-one-page)
5. [Headless training workflow](#headless-training-workflow)
6. [Playing back a trained policy](#playing-back-a-trained-policy)
7. [Task 1 — InvertedPendulum](#task-1--invertedpendulum-v5)
8. [Task 2 — InvertedDoublePendulum](#task-2--inverteddoublependulum-v5)
9. [Task 3 — Hopper](#task-3--hopper-v5)
10. [Task 4 — Ant at a constant speed](#task-4--antconstantspeed-v0-custom)
11. [Reading the training logs](#reading-the-training-logs)
12. [Where to go next](#where-to-go-next)

---

## Quick start

```bash
source ~/workspaces/.venv/bin/activate
cd ~/workspaces/projects/mujoco-rl

python scripts/train_inverted_pendulum.py              # train (≈1 min), also writes a video
python scripts/watch.py --model models/InvertedPendulum-v5_ppo.zip              # watch live
python scripts/watch.py --model models/InvertedPendulum-v5_ppo.zip --mode video # or render to mp4
```

To train **every** scenario one after another (≈5–6 h total), headless:

```bash
nohup bash scripts/train_all.sh > logs/train_all.out 2>&1 &
tail -f logs/train_all.out                      # one line as each task starts/finishes
bash scripts/train_all.sh hopper ant_constant_speed   # or just some of them
```

The Python venv lives at `~/workspaces/.venv` and is shared across projects.
`requirements.txt` pins the versions this repo was built against
(`gymnasium[mujoco]`, `mujoco`, `stable-baselines3`, `torch`, `tensorboard`,
`imageio`).

---

## Project layout

```
scripts/
  common.py                          shared setup + train()/play() helpers. IMPORT FIRST.
  ant_constant_speed.py              custom env: Ant rewarded for walking at a target speed
  train_inverted_pendulum.py         task 1
  train_inverted_double_pendulum.py  task 2
  train_hopper.py                    task 3
  train_ant_constant_speed.py        task 4
  train_all.sh                       train all four tasks in sequence
  train.py                           generic trainer for any env id (untuned defaults)
  watch.py                           play back any saved policy (live window or .mp4)
  pick_best.py                       score every checkpoint, promote the best to <run>.zip
models/                              trained policies, one set of files per run:
  <run>.zip                            policy + value network weights
  <run>_vecnormalize.pkl               observation-normalization statistics
  <run>_config.json                    env id, env settings, algorithm (used by watch.py)
  <run>_checkpoints/                   periodic snapshots (Hopper/Ant: every 100k steps)
logs/                                TensorBoard logs, one folder per training run
videos/                              rendered .mp4 playbacks
```

`<run>` defaults to `<env id>_<algo>`, e.g. `Hopper-v5_sac`. Pass `--run-name`
to any training script to keep several runs side by side, for example to
compare seeds.

---

## The segfault fix

**Symptom:** a script crashes with `Segmentation fault (core dumped)` the
moment Stable-Baselines3 builds its model, with no Python traceback.

**Root cause:** the shared venv has TensorFlow 2.21 installed. This project
doesn't use it, and it isn't in `requirements.txt`; another project brought it
in. The chain of events is:

1. SB3's TensorBoard logger imports `tensorboard`.
2. `tensorboard` imports TensorFlow if it can find it, which loads TF's
   native libraries.
3. Creating the first torch `Adam` optimizer lazily imports `torch._dynamo`,
   which loads **Triton**'s native library (`triton/knobs.py` →
   `triton._C.libtriton`).
4. Triton's native code segfaults if TensorFlow's native code is already
   loaded in the same process. Importing them in the opposite order (Triton
   first) works fine, which is why the old workaround of putting
   `import torch._dynamo` at the top of each script happened to help.

**The fix** lives in `scripts/common.py`, so every script gets it by importing
that module first:

```python
sys.modules["tensorflow"] = None   # `import tensorflow` now raises ImportError
```

This makes TensorFlow un-importable *for this process only*. TensorBoard
notices TF is missing and uses its own built-in TF-free event writer, so
logging works exactly as before. Startup is also a few seconds faster. Nothing
is uninstalled, so other projects that share the venv are unaffected.

**Rule for new scripts:** make `import common` the first import, before
gymnasium, mujoco, torch, or stable_baselines3:

```python
import common  # noqa: F401  (must be first: segfault fix + headless rendering)
```

(An alternative permanent fix is `pip uninstall tensorflow` from the shared
venv, but only if no other project needs it.)

`common.py` also handles headless rendering. If there's no `$DISPLAY`, it sets
`MUJOCO_GL=egl` so MuJoCo renders on the GPU without a window, which means
video recording works over SSH too.

---

## RL in one page

### The loop

Every Gymnasium environment exposes the same interface:

```python
obs, info = env.reset()
while True:
    action = policy(obs)                                        # agent decides
    obs, reward, terminated, truncated, info = env.step(action) # physics advances
    if terminated or truncated:
        obs, info = env.reset()
```

- **Observation** (`obs`): a vector of numbers describing the robot's current
  state, such as joint angles, joint velocities, body height, and sometimes
  contact forces. It's all the agent knows about the world.
- **Action**: a vector of continuous motor commands in `[-1, 1]` (or `[-3, 3]`
  for the single pendulum), one per actuator. MuJoCo multiplies each by the
  motor's `gear` ratio to get a force or torque. One `env.step` holds the
  action for `frame_skip` physics substeps. The resulting control period is
  the env's `dt`, listed in each task's section.
- **Reward**: one number per step saying how good that step was. Its design
  **defines the task**. The agent optimizes exactly what the reward says, not
  what you meant (task 4 is an example).
- **Episode**: one attempt. It ends because the robot failed (fell over,
  tipped too far), which is `terminated`, or because a time limit was hit (1000
  steps for every task here), which is `truncated`. The difference matters:
  after a termination there is no future reward, but after truncation the
  episode *could* have continued. The algorithms account for this.
- **Return**: the sum of rewards over an episode. This is the number in the
  training logs (`ep_rew_mean`) and the one `watch.py` prints.

### The policy

The **policy** is a small neural network (an MLP) mapping
`observation → action`. During training it's *stochastic*: it outputs the mean
and standard deviation of a Gaussian, and the action is sampled from it, which
gives the exploration noise RL needs to discover anything new. At playback we
use the **deterministic** policy, meaning just the mean action.

Both algorithms also train a **critic** (value network) that predicts how much
future reward to expect from a state (or from a state-action pair). The critic
is only a training aid: it tells the policy which actions turned out better
than expected. It isn't used at playback.

### The two algorithms

| | **PPO** (Proximal Policy Optimization) | **SAC** (Soft Actor-Critic) |
|---|---|---|
| Type | on-policy | off-policy |
| Learns from | fresh rollouts from the current policy; data is discarded after each update | a replay buffer of the last 1M transitions, reused many times |
| Sample efficiency | low (needs more simulation) | high |
| Stability | very stable, easy to tune | stable; learns its own exploration level |
| Default network (SB3) | actor 64-64 tanh, critic 64-64 tanh | actor 256-256 ReLU, twin Q-critics 256-256 ReLU |
| Action distribution | Gaussian, clipped to bounds | Gaussian squashed through `tanh` into bounds |
| Used here for | single pendulum (cheap to simulate, simple) | double pendulum, Hopper, Ant |

**PPO** collects a batch of experience (`n_steps × n_envs` transitions), uses
the critic to estimate each action's *advantage* (how much better it was than
average, via GAE), and nudges the policy toward high-advantage actions. The
"proximal" part clips each update so the policy can't change too much at once,
which is where its stability comes from.

**SAC** stores every transition in a replay buffer and, after each env step,
samples a minibatch of 256 to update. It learns two Q-functions (critics) and
uses the smaller of the two to avoid over-optimism. Its objective is *reward +
α·entropy*: it's rewarded for staying random where randomness doesn't hurt,
and `ent_coef="auto"` tunes α automatically. That built-in exploration is a
big reason it works so well on locomotion.

### Observation normalization

Every script wraps the env in SB3's `VecNormalize`, which keeps a running mean
and standard deviation of every observation dimension and feeds the network
`(obs - mean) / std`. MuJoCo observations mix wildly different scales: angles
around 0.1, velocities around 10, contact forces in the 100s. Networks learn
much better when every input is roughly unit-scale. PPO runs also normalize
rewards. SAC runs don't, because SAC's entropy tuning handles reward scale.

**Consequence:** the policy only understands *normalized* observations. Those
statistics are saved to `<run>_vecnormalize.pkl` and must be reloaded,
**frozen**, at playback. `watch.py` does this automatically. Forgetting it is
one of the most common RL bugs: the policy appears to have "forgotten"
everything.

---

## Headless training workflow

The training scripts never open a window. A typical session on a remote
machine, or for a long run you don't want to babysit, looks like this:

```bash
source ~/workspaces/.venv/bin/activate
cd ~/workspaces/projects/mujoco-rl

# 1. start training detached from the terminal; survives logout
nohup python scripts/train_hopper.py > logs/hopper.out 2>&1 &
#    (or run it inside `tmux new -s hopper` and detach with Ctrl-b d)

# 2. check on it
tail -f logs/hopper.out          # SB3 prints a stats table every few thousand steps
tensorboard --logdir logs        # learning curves at http://localhost:6006
#    remote machine? forward the port from your laptop:  ssh -L 6006:localhost:6006 <host>

# 3. intermediate snapshots appear while it trains (Hopper/Ant: every 100k steps)
ls models/Hopper-v5_sac_checkpoints/
```

Options shared by every `train_*.py` script:

| Flag | Meaning |
|---|---|
| `--timesteps N` | total env steps to train for |
| `--n-envs N` | parallel simulator copies used to collect experience |
| `--seed N` | random seed; RL results vary a lot between seeds, so try 2–3 |
| `--run-name NAME` | file prefix under `models/`, `logs/`, `videos/` |
| `--checkpoint-every N` | save a snapshot every N steps (Hopper/Ant default 100k) |
| `--device cuda` | train on GPU; usually *slower* for these tiny MLPs |
| `--no-video` | skip the automatic video at the end |

When training finishes, the script saves the policy, normalization stats, and
config, then **automatically renders one episode to `videos/<run>.mp4`**
(headless, through EGL), so you can check the result without a display.

---

## Playing back a trained policy

`scripts/watch.py` loads any saved policy. It reads the env and algorithm
from `<run>_config.json` and the normalization stats from
`<run>_vecnormalize.pkl`, so the only required argument is the model path.

```bash
# live, interactive MuJoCo window (needs a display: at the desktop, or ssh -X)
python scripts/watch.py --model models/Hopper-v5_sac.zip

# headless: render to an mp4 (works anywhere), default path videos/<model>.mp4
python scripts/watch.py --model models/Hopper-v5_sac.zip --mode video --episodes 3

# custom output path
python scripts/watch.py --model models/Hopper-v5_sac.zip --mode video --video-path ~/hopper.mp4

# watch an intermediate checkpoint, to see how behaviour evolved during training
python scripts/watch.py --model models/Hopper-v5_sac_checkpoints/Hopper-v5_sac_300000_steps.zip --mode video

# sample actions with exploration noise instead of the mean action
python scripts/watch.py --model models/Hopper-v5_sac.zip --stochastic
```

For every episode it prints the return, the length in steps, and the mean
forward speed for locomotion tasks:

```
Episode 1: return =  3412.55   length = 1000   mean forward speed = 3.21 m/s
```

Videos play in real time (fps = 1 / env `dt`). To view a video rendered on a
remote machine, copy it back with `scp <host>:~/workspaces/projects/mujoco-rl/videos/*.mp4 .`.

### Picking the best checkpoint

Off-policy algorithms like SAC don't improve monotonically, and the policy at
the very end of training is often *not* the best one. Hopper in this repo is a
real example: its 900k-step checkpoint completed 10/10 test episodes, while
the final 1M-step policy completed only 3/10. After a SAC run, score every
checkpoint and promote the best:

```bash
python scripts/pick_best.py --run Hopper-v5_sac              # 10 deterministic episodes each
python scripts/pick_best.py --run Hopper-v5_sac --dry-run    # just print the scores
```

The winner is copied to `models/<run>.zip`, together with its matching
normalization stats. The original end-of-training policy is kept as
`models/<run>_final.zip`.

In the live window you can drag to rotate the camera, right-drag to pan,
scroll to zoom, and press Space to pause.

---

## Task 1 — `InvertedPendulum-v5`

**What it is.** A cart slides along a rail, with a pole hinged on top. Push
the cart left and right to keep the pole upright. It's the "hello world" of
continuous control and a sanity check that the pipeline works.

```bash
python scripts/train_inverted_pendulum.py
python scripts/watch.py --model models/InvertedPendulum-v5_ppo.zip
```

**Observation**: 4 numbers.

| idx | meaning | unit |
|---|---|---|
| 0 | cart position along the rail | m |
| 1 | pole angle from vertical | rad |
| 2 | cart velocity | m/s |
| 3 | pole angular velocity | rad/s |

**Action**: 1 number in `[-3, 3]`, the force pushing the cart (× gear 100).
Control period `dt` = 0.04 s (25 Hz).

**Reward**: `+1` for every step the pole is still up. That's all: "survive as
long as possible".

**Termination**: `|pole angle| > 0.2 rad` (≈11°), or a non-finite state.
**Truncation**: 1000 steps (40 s). **Maximum return: 1000.**

**Policy trained**: PPO, actor and critic each a 64-64 tanh MLP, SB3 defaults
(`n_steps=2048` per env × 4 envs, `batch_size=64`, 10 epochs per update,
`lr=3e-4`, `γ=0.99`, `λ=0.95`, clip 0.2). 100k steps.

**Result achieved here**: return 1000/1000 (balances for the whole episode) in
about 1 minute of training.

**What to notice**: the reward never says *how* to balance. The agent learns
that the cart must move *under* the falling pole, a non-obvious control law,
purely from "you lasted longer when you did that".

---

## Task 2 — `InvertedDoublePendulum-v5`

**What it is.** The same cart, but with *two* poles hinged end to end. It's a
chaotic system: small errors amplify within a fraction of a second, and the
controller has to move the cart to keep the *top* tip above the cart.

```bash
python scripts/train_inverted_double_pendulum.py
python scripts/watch.py --model models/InvertedDoublePendulum-v5_sac.zip
```

**Observation**: 9 numbers.

| idx | meaning |
|---|---|
| 0 | cart position (m) |
| 1, 2 | sin of pole-1 angle, sin of pole-2 angle (relative to pole 1) |
| 3, 4 | cos of the same two angles |
| 5 | cart velocity (clipped ±10) |
| 6, 7 | pole-1 and pole-2 angular velocities (clipped ±10) |
| 8 | constraint force on the cart's rail (clipped ±10) |

Angles come in as sin/cos pairs rather than raw radians, so the input is
continuous across wrap-around (θ and θ+2π look identical).

**Action**: 1 number in `[-1, 1]`, the cart force (× gear 500).
`dt` = 0.05 s (20 Hz).

**Reward** per step:

```
reward = 10                                   # alive bonus
       - (0.01 * tip_x² + (tip_y - 2)²)       # distance penalty: keep the tip high (2 m = fully upright) and near the centre
       - (1e-3 * ω1² + 5e-3 * ω2²)            # velocity penalty: don't swing the poles fast
```

`tip_x` and `tip_y` are the world position of the free end of the top pole.

**Termination**: tip height `tip_y ≤ 1 m` (the poles have folded).
**Truncation**: 1000 steps. **Maximum return ≈ 9350.**

**Policy trained**: SAC, actor and twin critics 256-256 ReLU, SB3 defaults
(buffer 1M, batch 256, `lr=3e-4`, `γ=0.99`, `τ=0.005`, auto entropy) with a
10k-step random warm-up. 200k steps, checkpoint every 50k.

**Why SAC and not PPO here** (a lesson learned building this repo): PPO with
the rl-baselines3-zoo tuned hyperparameters reached a training return of about
7,700 after 1M steps. In playback, though, it balanced from only about 75–90%
of start states, and the rest fell within ~10–20 steps. SAC with defaults
balanced **20/20** test episodes at the maximum return (≈9,360) after 200k
steps. SAC's replay buffer keeps the rare, hard starting states around and
learns from them again and again, where PPO uses each rollout once and throws
it away.

**What to notice**: the dense shaping terms (distance and velocity penalties)
give the agent a gradient to follow *before* it can balance for long. Compare
this with task 1's pure survival reward.

---

## Task 3 — `Hopper-v5`

**What it is.** A 2D one-legged robot made of a torso, thigh, shin, and foot.
It's constrained to a vertical plane and must hop forward without falling. This
is the first real **locomotion** task: no gait is given, so the agent must
invent hopping from the reward alone.

```bash
python scripts/train_hopper.py                     # 1M steps
python scripts/watch.py --model models/Hopper-v5_sac.zip
```

**Observation**: 11 numbers. Joint positions come first, then velocities.
The torso's x position is excluded so the policy behaves the same wherever it
is on the track.

| idx | meaning |
|---|---|
| 0 | torso height z (m) |
| 1 | torso pitch angle (rad) |
| 2 | thigh joint angle |
| 3 | leg (knee) joint angle |
| 4 | foot joint angle |
| 5 | torso x velocity (forward speed) |
| 6 | torso z velocity |
| 7 | torso pitch angular velocity |
| 8–10 | thigh, leg, and foot joint angular velocities |

Velocities are clipped to ±10.

**Action**: 3 numbers in `[-1, 1]`, torques for the thigh, leg, and foot
motors (× gear 200). `dt` = 0.008 s (125 Hz).

**Reward** per step:

```
reward = x_velocity              # forward reward: Δx / dt, positive when moving forward
       + 1                       # healthy reward, while not fallen
       - 1e-3 * Σ action²        # control cost: discourages wasteful max-torque flailing
```

**Termination** (fell over / unhealthy): torso height `< 0.7 m`, `|torso
angle| > 0.2 rad`, or any state value `> 100` in magnitude.
**Truncation**: 1000 steps (8 s).
**Good return**: about 3,000 or more. A random policy gets about 15–20.

**Policy trained**: SAC, actor and twin critics each 256-256 ReLU, with
`learning_starts=10,000` (pure random actions first, to fill the replay buffer
with diverse data), `buffer_size=1M`, `batch_size=256`, `lr=3e-4`, `γ=0.99`,
`τ=0.005` (soft target update), and automatic entropy tuning. Observations are
normalized; rewards are not. 1M steps, checkpoint every 100k.

**Result achieved here** (1M steps, 126 min): the best checkpoint (900k steps,
promoted with `pick_best.py`) completes **10/10** episodes, with a return of
about **3,650**, hopping steadily at **2.65 m/s**. The final 1M-step policy
was worse (3/10 completed, about 2,800), a typical late-training SAC
oscillation.

**What to notice**: watch the checkpoints in order. The agent usually learns
to *stand* first, because the +1 healthy reward is easy money. It then learns
to lean and fall forward (a burst of `x_velocity`), and only then to hop
repeatedly. A common local optimum is a policy that just stands still
collecting +1 per step, for a return of about 1000. If a run plateaus there,
try another seed.

---

## Task 4 — `AntConstantSpeed-v0` (custom)

**What it is.** A 3D four-legged "ant" with a torso and four legs, each with a
hip and an ankle joint. The standard `Ant-v5` rewards forward speed without
limit, so trained ants sprint and lunge. This repo defines a variant that
rewards walking **at a target speed** (default 1.0 m/s), in a straight line.
The definition is in `scripts/ant_constant_speed.py`.

```bash
python scripts/train_ant_constant_speed.py                     # 1 m/s, 1M steps
python scripts/train_ant_constant_speed.py --target-speed 0.5  # slower walk -> models/AntConstantSpeed-v0_0.5ms_sac.zip
python scripts/watch.py --model models/AntConstantSpeed-v0_sac.zip
```

**Observation**: 105 numbers.

| idx | meaning |
|---|---|
| 0 | torso height z (m) |
| 1–4 | torso orientation as a quaternion (w, x, y, z) |
| 5–12 | 8 joint angles: hip and ankle for each of the 4 legs |
| 13–15 | torso linear velocity (x, y, z), including the current forward speed |
| 16–18 | torso angular velocity |
| 19–26 | 8 joint angular velocities |
| 27–104 | external contact forces/torques on 13 bodies × 6, clipped to [-1, 1] |

Torso x and y position are excluded, so the policy is the same anywhere on the
floor. The target speed is *not* in the observation, because it's fixed for a
given trained policy. To make one policy follow a changing speed command, you
would append the command to the observation (see "where to go next").

**Action**: 8 numbers in `[-1, 1]`, torques for the 4 hips and 4 ankles
(× gear 150). `dt` = 0.05 s (20 Hz).

**Reward** per step:

```
tracking = exp( -(v_x - v_target)² / 0.5² )      # 1.0 exactly at the target speed, ~0.37 at ±0.5 m/s, ~0.02 at ±1 m/s

reward = 2.0 * tracking                          # walk at the target speed
       - 0.5 * v_y²                              # walk straight: penalize sideways drift
       + 1                                       # healthy reward (from Ant-v5)
       - 0.5 * Σ action²                         # control cost (from Ant-v5)
       - 5e-4 * Σ F_contact²                     # contact cost (from Ant-v5), discourages slamming into the ground
```

The design choices:

- **A Gaussian "tracking" kernel** in place of Ant-v5's `+v_x` term. Its
  reward peaks at the target and falls off whether the ant is too slow *or*
  too fast, so sprinting stops paying. This kernel is the standard choice in
  modern legged-robot RL (for example ETH's `legged_gym` for ANYmal).
- **σ = 0.5 m/s** sets how forgiving the reward is. A smaller σ demands more
  precise speed but gives almost no learning signal when far from the target,
  for example when standing still at the start.
- **Weight 2.0 vs the +1 healthy bonus**: walking at the target is worth more
  than just surviving, so "stand still forever" (about 1 per step) is clearly
  worse than "walk at 1 m/s" (about 3 per step, minus costs).
- **Lateral penalty**: without it, drifting diagonally at 1 m/s overall would
  still earn some tracking reward from its x component.

`info` also reports `reward_tracking`, `reward_lateral`, and `speed_error`
each step, so you can see which term dominates.

**Termination**: torso height outside `[0.2, 1.0] m` (flipped over or jumped).
**Truncation**: 1000 steps (50 s).
**Good result**: `watch.py` reports a mean forward speed within about 0.1 m/s
of the target, with returns of roughly 2,500 or more.

**Policy trained**: SAC with the same settings as Hopper (256-256 ReLU,
`learning_starts=10k`, buffer 1M, batch 256, `lr=3e-4`, `γ=0.99`, `τ=0.005`,
auto entropy). 1M steps, checkpoint every 100k.

**What to notice**: this is reward *design*, the hardest practical skill in
RL. Try changing `tracking_sigma`, `tracking_weight`, or `lateral_weight` in
`ant_constant_speed.py`, retrain with a different `--run-name`, and compare
the gaits. A large control cost gives a cautious shuffle; a tiny one gives
twitchy, energetic motion.

---

## Reading the training logs

SB3 prints a table every few thousand steps, and TensorBoard
(`tensorboard --logdir logs`) plots the same values:

| Metric | Meaning | What you want |
|---|---|---|
| `rollout/ep_rew_mean` | mean return of the last 100 training episodes | goes up, then plateaus near the task's max |
| `rollout/ep_len_mean` | mean episode length | rises to 1000 when the agent stops failing |
| `train/entropy_loss` (PPO) / `train/ent_coef` (SAC) | how random the policy still is | slowly decreasing |
| `train/value_loss` (PPO) / `train/critic_loss` (SAC) | critic prediction error | no strict target; spikes when behaviour changes |
| `train/approx_kl`, `train/clip_fraction` (PPO) | how much each update changes the policy | small (KL ≲ 0.05); large means unstable |
| `time/fps` | env steps per second | higher is faster training |

Training-time returns use the *stochastic* policy. Deterministic playback in
`watch.py` is usually a bit better, but can occasionally be worse.

---

## Where to go next

- **More tasks with the generic trainer**: `python scripts/train.py --env
  Walker2d-v5 --algo sac --timesteps 1000000`. Other good ones are
  `HalfCheetah-v5` and `Ant-v5` (the sprinting version), then `Humanoid-v5`
  (5–10M steps).
- **A commanded-speed ant**: sample a new `target_speed` at every `reset()`,
  append it to the observation, and train one policy that walks at any
  requested speed. This is how real legged robots are trained.
- **Robustness**: randomize masses, friction, or push the robot mid-episode
  (domain randomization), so the policy survives perturbations.
- **Compare algorithms**: train the double pendulum with
  `python scripts/train.py --env InvertedDoublePendulum-v5 --algo ppo --timesteps 1000000 --n-envs 8`
  and overlay its curve with the SAC run in TensorBoard.
