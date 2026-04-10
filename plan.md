# Vision-Based RL Navigation in Simulation
## Implementation Plan and Roadmap

## 1. Project Overview

Build a **purely simulation-based vision RL project** in **PyTorch** using **PufferLib** where an agent learns to navigate to a goal using only visual observations.

### Core idea
- Agent sees an RGB image from a simulated first-person camera
- Agent outputs discrete navigation actions
- Environment is procedurally generated
- Training is done with PPO via PufferLib

### Target outcome
A trained policy that can navigate unseen simulated mazes with a reasonable success rate using only visual input.

---

## 2. Objectives

### Primary objectives
- Build a custom simulation environment for visual navigation
- Train an RL agent using **PufferLib + PyTorch**
- Use a **CNN-based policy** on image observations
- Evaluate generalization on unseen maps

### Secondary objectives
- Add curriculum learning
- Add memory with LSTM
- Improve robustness with domain randomization

---

## 3. Scope

### In scope
- Simulation-only environment
- RGB visual input
- Discrete action space
- PPO training
- Evaluation metrics and visualizations
- Reproducible training pipeline

### Out of scope
- Real robot deployment
- Real-world image datasets
- Continuous-control drones with complex physics
- Multi-agent coordination in v1

---

## 4. Technical Stack

- **Python**
- **PyTorch**
- **PufferLib**
- **Gymnasium-compatible custom environment**
- **NumPy**
- **OpenCV / PyGame / simple renderer**
- **Matplotlib** for logs and plots

---

## 5. Environment Specification

## Task
The agent starts in a random map and must reach a goal while avoiding walls and obstacles.

## Observation space
- RGB image
- Shape: `(3, 64, 64)` or `(3, 84, 84)`
- First-person camera view

## Action space
Discrete actions:
- `0`: move forward
- `1`: turn left
- `2`: turn right
- `3`: stay still (optional)

## Reward design
- `+10` for reaching goal
- `-5` for collision
- `-0.01` per step
- `+small progress reward` when distance to goal decreases

## Termination conditions
- Goal reached
- Collision
- Max steps exceeded

---

## 6. Model Architecture

## Baseline model
CNN-based actor-critic model:
- Convolutional encoder
- Shared latent representation
- Policy head
- Value head

## Suggested architecture
- Conv(32 filters)
- Conv(64 filters)
- Conv(64 filters)
- Flatten
- Linear(512)
- Policy logits head
- Value head

## Future upgrade
- CNN + LSTM for partial observability

---

## 7. Training Plan

## RL algorithm
- PPO through **PufferLib**

## Training setup
- Parallel environments for sample efficiency
- Randomized map generation
- Fixed training/eval seeds for reproducibility

## Initial hyperparameters
- Learning rate: `3e-4`
- Discount factor: `0.99`
- GAE lambda: `0.95`
- PPO clip: `0.2`
- Entropy coefficient: `0.01`
- Batch size: tune based on hardware
- Number of environments: `32` to `256`

---

## 8. Evaluation Metrics

Track the following:
- Success rate
- Average episode reward
- Average episode length
- Collision rate
- Distance-to-goal reduction
- Generalization performance on unseen maps

## Validation protocol
- Train on one distribution of maps
- Evaluate on held-out random seeds
- Compare across difficulty levels

---

## 9. Project Structure

```text
project/
├── configs/
│   ├── train.yaml
│   ├── env.yaml
│   └── model.yaml
├── envs/
│   ├── navigation_env.py
│   ├── map_generator.py
│   └── renderer.py
├── models/
│   ├── cnn_policy.py
│   └── cnn_lstm_policy.py
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── callbacks.py
├── utils/
│   ├── metrics.py
│   ├── visualization.py
│   └── seeding.py
├── results/
├── videos/
├── notebooks/
├── README.md
└── requirements.txt
````

---

## 10. Implementation Phases

## Phase 1: Environment Prototype

### Goal

Create a minimal working simulation environment.

### Tasks

* Define observation and action spaces
* Implement agent movement
* Implement wall collision logic
* Implement goal placement
* Add reward and termination logic
* Add rendering pipeline for RGB observations

### Deliverable

A Gym-compatible environment that can be stepped manually.

### Exit criteria

* Environment runs without training
* Random policy can interact with it
* Episode loop works correctly

---

## Phase 2: PufferLib Integration

### Goal

Connect the custom environment to PufferLib.

### Tasks

* Wrap environment for PufferLib usage
* Verify vectorized environment execution
* Implement config handling for training
* Add logging hooks

### Deliverable

PufferLib can launch rollouts on multiple environments in parallel.

### Exit criteria

* PPO training loop starts and runs
* Observations/actions are validated
* No environment shape/runtime errors

---

## Phase 3: Baseline Vision Policy

### Goal

Train a CNN policy on simple maps.

### Tasks

* Implement CNN actor-critic in PyTorch
* Normalize observations
* Train on small/simple mazes
* Plot reward curves and success rate

### Deliverable

A baseline model that learns basic navigation.

### Exit criteria

* Learning curve improves over random baseline
* Agent reaches goal in simple maps consistently

---

## Phase 4: Procedural Map Generation

### Goal

Increase task diversity and reduce overfitting.

### Tasks

* Build random maze/map generator
* Randomize spawn and goal positions
* Add different obstacle layouts
* Add difficulty parameter

### Deliverable

A procedurally generated environment family.

### Exit criteria

* Agent trains across multiple layouts
* Performance does not collapse under moderate variation

---

## Phase 5: Evaluation and Generalization

### Goal

Measure how well the agent generalizes.

### Tasks

* Create held-out test maps
* Measure success rate across difficulties
* Save rollout videos
* Analyze failure cases

### Deliverable

Evaluation report with plots and examples.

### Exit criteria

* Reproducible evaluation script
* Clear benchmark results on unseen maps

---

## Phase 6: Robustness Improvements

### Goal

Improve stability and realism.

### Tasks

* Add curriculum learning
* Add texture/color randomization
* Tune reward shaping
* Tune PPO hyperparameters

### Deliverable

Improved training stability and stronger generalization.

### Exit criteria

* Higher eval success rate
* Reduced collision rate
* Better transfer to unseen map styles

---

## Phase 7: Memory-Based Extension

### Goal

Handle partial observability better.

### Tasks

* Replace CNN policy with CNN + LSTM
* Track recurrent hidden states
* Compare against feedforward baseline

### Deliverable

A memory-enabled navigation agent.

### Exit criteria

* LSTM outperforms CNN baseline on harder maps
* Agent handles long corridors and hidden turns better

---

## 11. Roadmap

## Week 1

* Finalize project spec
* Set up repo and dependencies
* Build minimal environment
* Add manual controls and rendering

## Week 2

* Complete rewards, resets, collision logic
* Validate observation pipeline
* Integrate with PufferLib

## Week 3

* Implement CNN actor-critic
* Run first PPO training
* Debug rollout and training issues

## Week 4

* Train baseline on simple maps
* Save checkpoints
* Plot reward and success curves

## Week 5

* Add procedural generation
* Expand difficulty range
* Train on randomized maps

## Week 6

* Build evaluation suite
* Test on unseen environments
* Record rollout videos

## Week 7

* Add domain randomization and curriculum learning
* Tune hyperparameters
* Improve stability

## Week 8

* Add CNN + LSTM extension
* Compare against baseline
* Write final report and README

---

## 12. Milestones

### Milestone 1

Environment is fully playable and stable.

### Milestone 2

PufferLib PPO successfully trains on the environment.

### Milestone 3

CNN policy solves simple navigation reliably.

### Milestone 4

Policy generalizes to unseen procedural maps.

### Milestone 5

LSTM extension improves performance on harder tasks.

---

## 13. Risks and Mitigation

## Risk 1: Training instability

**Mitigation**

* Start with very simple maps
* Keep rewards dense enough
* Tune PPO clip, entropy, and learning rate

## Risk 2: Visual observations too hard too early

**Mitigation**

* Start with low-noise simple textures
* Use curriculum learning
* Add domain randomization later

## Risk 3: Environment bugs

**Mitigation**

* Test environment separately before training
* Add unit checks for reset, step, reward, done logic

## Risk 4: Overfitting to map layouts

**Mitigation**

* Use procedural generation
* Hold out evaluation seeds
* Randomize spawn/goal locations

---

## 14. Final Deliverables

* Custom visual navigation environment
* PufferLib-compatible training pipeline
* CNN PPO baseline in PyTorch
* Evaluation script and metrics
* Saved checkpoints
* Demo rollout videos
* README with setup, training, and results
* Final report with observations and future work

---

## 15. Success Criteria

The project is considered successful if:

* The agent learns to navigate in simulation from visual input
* Training is reproducible
* The model performs better than random on unseen maps
* Results are documented with plots, metrics, and videos

---

## 16. Stretch Goals

* Continuous control variant
* Depth-map input instead of RGB
* Multi-goal navigation
* Dynamic obstacles
* Imitation learning warm-start
* Transformer-based visual encoder

---


