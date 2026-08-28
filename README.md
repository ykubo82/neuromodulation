# Neuromodulated Equilibrium Learning (NEL)

This repository contains PyTorch implementations of **Neuromodulated Equilibrium Learning (NEL)** and the **Direct Cost Feedback (DCF)** control used in the accompanying paper.

NEL is a single-equilibrium, reward-modulated learning framework. The network first relaxes to a free equilibrium, selects an action/class using a Max-Boltzmann policy, receives a binary reward, and propagates a reward-dependent modulatory signal through reciprocal feedback connections to drive local synaptic updates.

NEL does **not** use:

- a reward-nudged equilibrium,
- a contrastive Equilibrium Propagation update, or
- activation derivatives in the modulatory feedback pathway.

DCF uses the same free-equilibrium dynamics and local feedback-based update structure as NEL, but replaces the sparse selected-action reward signal with the full supervised output signal.

## Files

- `main_nel_general.py` — NEL for multilayer perceptrons (MLPs) on MNIST and Fashion-MNIST.
- `main_nel_cnn.py` — NEL for convolutional neural networks (CNNs), including the CIFAR-10 experiments.
- `main_direct_cost_feedback_mlp.py` — DCF control for MLPs.
- `main_direct_cost_feedback_cnn.py` — DCF control for CNNs.

## Method Summary

For an input sample, NEL first performs synchronous discrete-time Euler relaxation until the network reaches a free equilibrium.

For hidden layer `l`:

```text
h_l(t+1) = h_l(t)
           + dt * [
               -h_l(t)
               + rho_l(
                   W_l h_{l-1}(t)
                   + gamma * W_{l+1}^T h_{l+1}(t)
                   + b_l
               )
           ]
```

The output state evolves according to:

```text
y(t+1) = y(t)
         + dt * [
             -y(t)
             + rho_out(
                 W_out h_last(t)
                 + b_out
             )
         ]
```

All states on the right-hand side are evaluated at the same time step, so the updates are synchronous.

After free relaxation, a class/action `a` is selected. With probability `1 - epsilon`, the greedy class is used. With probability `epsilon`, an exploratory action is sampled from:

```text
a ~ Categorical(logits = y)
```

The binary reward is:

```text
r = 1  if a is the target class
r = 0  otherwise
```

The output-layer modulatory signal is:

```text
m_L = (r - y_a) e_a
```

Hidden-layer modulators are propagated through reciprocal transpose feedback connections:

```text
m_l = W_{l+1}^T m_{l+1}
```

The local synaptic update is:

```text
Delta W_l = (eta_l / B) * sum_n [
    (m_{l,n} * h_{l,n}) h_{l-1,n}^T
]
```

Here, `*` inside the parenthesis denotes element-wise multiplication.

All layer-wise updates are computed before any parameter is modified.

### Direct Cost Feedback Control

DCF uses the same free-equilibrium dynamics, reciprocal feedback pathway, and local update rule as NEL, but replaces the sparse reward-based output signal with the full supervised output signal:

```text
m_L^DCF = t - y
```

where `t` is the one-hot target vector and `y` is the free-equilibrium output.

DCF uses no action selection, exploration, reward signal, reward-nudged phase, or contrastive update.

## Requirements

The code requires Python 3 and PyTorch.

A minimal environment is:

```bash
pip install torch torchvision numpy
```

GPU execution is used automatically when CUDA is available.

## Datasets

The scripts download datasets automatically through `torchvision`.

The preprocessing used in the paper is:

| Dataset | Mean | Standard deviation |
|---|---|---|
| MNIST | `(0.1307,)` | `(0.3081,)` |
| Fashion-MNIST | `(0.2860,)` | `(0.3530,)` |
| CIFAR-10 | `(0.4914, 0.4822, 0.4465)` | `(0.2470, 0.2435, 0.2616)` |

No data augmentation is used in the reported CIFAR-10 experiments.

## Reproducing the Main NEL Experiments

The commands below reproduce the principal NEL configurations reported in the paper. Change `--seed` to run additional random seeds.

### MNIST — 784-512-10 MLP

```bash
python main_nel_general.py --task MNIST --archi 784 512 10 --T1 80 --dt 0.1 --gamma 1.0 --exploration 0.2 --act relu --output-act hard_sigmoid --lrs 0.1 0.1 --mbs 64 --epochs 80 --seed 1 --save
```

### Fashion-MNIST — 784-1024-10 MLP

```bash
python main_nel_general.py --task FMNIST --archi 784 1024 10 --T1 80 --dt 0.1 --gamma 1.0 --exploration 0.2 --act relu --output-act hard_sigmoid --lrs 0.1 0.1 --mbs 64 --epochs 80 --seed 1 --save
```

### Fashion-MNIST — 784-512-256-10 MLP

This experiment uses bounded LeakyReLU hidden activities in `[-1.2, 1.2]`.

```bash
python main_nel_general.py --task FMNIST --archi 784 512 256 10 --T1 100 --dt 0.1 --gamma 1.0 --exploration 0.2 --act leaky_relu --act-clamp 1.2 --output-act hard_sigmoid --lrs 0.12 --mbs 64 --epochs 120 --seed 1 --save
```

### CIFAR-10 — Conv64-Conv128-10 CNN

This experiment uses bounded ReLU hidden activities in `[0, 1.2]`.

```bash
python main_nel_cnn.py --task CIFAR10 --channels 64 128 --kernel-size 5 --pool-size 2 --T1 80 --dt 0.2 --gamma 1.0 --exploration 0.2 --act relu --act-clamp 1.2 --output-act hard_sigmoid --lrs 0.02 0.02 0.02 --mbs 64 --epochs 120 --seed 1 --save
```

## Reproducing the Direct Cost Feedback Controls

### MNIST — 784-512-10 MLP

```bash
python main_direct_cost_feedback_mlp.py --task MNIST --archi 784 512 10 --T1 80 --dt 0.1 --gamma 1.0 --act relu --output-act hard_sigmoid --lrs 0.1 0.1 --mbs 64 --epochs 80 --seed 1 --save
```

### Fashion-MNIST — 784-1024-10 MLP

```bash
python main_direct_cost_feedback_mlp.py --task FMNIST --archi 784 1024 10 --T1 80 --dt 0.1 --gamma 1.0 --act relu --output-act hard_sigmoid --lrs 0.1 0.1 --mbs 64 --epochs 80 --seed 1 --save
```

### Fashion-MNIST — 784-512-256-10 MLP

```bash
python main_direct_cost_feedback_mlp.py --task FMNIST --archi 784 512 256 10 --T1 100 --dt 0.1 --gamma 1.0 --act leaky_relu --act-clamp 1.2 --output-act hard_sigmoid --lrs 0.08 --mbs 64 --epochs 120 --seed 1 --save
```

### CIFAR-10 — Conv64-Conv128-10 CNN

```bash
python main_direct_cost_feedback_cnn.py --task CIFAR10 --channels 64 128 --kernel-size 5 --pool-size 2 --T1 80 --dt 0.2 --gamma 1.0 --act relu --act-clamp 1.2 --output-act hard_sigmoid --lrs 0.02 0.02 0.02 --mbs 64 --epochs 120 --seed 1 --save
```

## Multiple Random Seeds

The main paper reports results over six random seeds. For example:

```bash
for seed in 1 2 3 4 5 6; do
    python main_nel_general.py --task MNIST --archi 784 512 10 --T1 80 --dt 0.1 --gamma 1.0 --exploration 0.2 --act relu --output-act hard_sigmoid --lrs 0.1 0.1 --mbs 64 --epochs 80 --seed ${seed} --save
done
```

## Saved Results

When `--save` is provided, each run creates a configuration-specific output directory.

The NEL scripts save:

- `train_acc.npy` — online training accuracy accumulated from free-equilibrium predictions during training.
- `test_acc.npy` — test accuracy evaluated after each completed epoch.
- `train_test_acc.npy` — stacked training/test accuracy histories.
- `config.json` — complete run configuration.
- `epoch_time.npy` — per-epoch runtime for the CNN implementation.

The first entry of the accuracy arrays is the chance-level reference value. Epoch 1 is therefore stored at index 1.

The reported **maximum test accuracy** for a seed is the maximum value observed in `test_acc.npy` over the fixed training duration.

## Important Note on `train_acc.npy`

`train_acc.npy` contains the online training accuracy accumulated over mini-batches during training. It is not a separate end-of-epoch evaluation of the final model on the complete training set.

If an analysis requires train and test accuracy evaluated using exactly the same fixed model state, the training set should be re-evaluated after each epoch using the same evaluation procedure as the test set.

## CNN Architecture

For the reported CIFAR-10 experiment, the architecture is:

```text
Input
  -> Conv(64, 5x5)
  -> MaxPool(2x2)
  -> Conv(128, 5x5)
  -> MaxPool(2x2)
  -> Linear(10)
```

There is no additional fully connected hidden layer.

The reciprocal pathway reuses the max-pooling switches for max-unpooling and applies transpose convolution with the tied forward kernels.

## Notes on Biological Plausibility

NEL removes the reward-nudged phase and activation derivatives from the modulatory feedback pathway. The current implementation still uses reciprocal feedback based on the transpose of the forward weights. Relaxing this weight-symmetry assumption is an important direction for future work.

## Paper

**Neuromodulated Equilibrium Learning**

Paper/preprint link will be added after public release.

## Citation

A BibTeX entry will be added after the paper is publicly available.

## License

Please add the license appropriate for your intended release.
