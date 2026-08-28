"""Neuromodulated Equilibrium Learning (NEL) for convolutional networks.

This implementation uses reciprocal convolutional equilibrium dynamics with
max-pooling, max-unpooling, and transpose-convolution feedback with tied
forward kernels. After a single free-equilibrium relaxation, a selected-action
reward signal is propagated through reciprocal feedback connections and used
to drive local synaptic updates.

There is no reward-nudged equilibrium, no contrastive EP update, and no
activation derivative in the NEL modulatory pathway.
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_weight
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# Neuromodulated Equilibrium Learning (NEL) for CNNs
# ============================================================
#
# This implementation combines:
#   1) the reciprocal convolutional equilibrium dynamics used in
#      Ernoult et al. (NeurIPS 2019), including max-pooling,
#      max-unpooling, and transpose-convolution feedback; and
#   2) the single-equilibrium NEL rule used in main_nel_general.py:
#
#         M_output = (r - y_a) e_a
#         M_lower  = transpose_feedback(M_upper)
#         dW       = correlation((M_post * post), pre)
#
# There is NO nudged phase and NO contrastive EP update.
# There is also NO activation derivative in the NEL modulator
# propagation. Max-pool switches are used only to transpose the
# pooling operation, analogously to the reciprocal CNN dynamics.
#
# Default CNN architecture mirrors the 2019 EP CNN topology:
#
#   1x28x28 -> Conv(32, 5x5) -> MaxPool(2)
#            -> Conv(64, 5x5) -> MaxPool(2)
#            -> Linear(10)
#
# For MNIST/FMNIST this gives pooled state sizes 12x12 and 4x4.
# ============================================================


# ============================================================
# Command-line arguments
# ============================================================


def build_parser():
    parser = argparse.ArgumentParser(
        description="Neuromodulated Equilibrium Learning (NEL) for convolutional networks"
    )

    parser.add_argument(
        "--task",
        type=str,
        default="MNIST",
        choices=["MNIST", "FMNIST", "CIFAR10"],
        help="Dataset",
    )

    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        default=[32, 64],
        help=(
            "Bottom-up convolutional channel sizes. "
            "Example: --channels 32 64 means input -> 32 -> 64."
        ),
    )

    parser.add_argument(
        "--fc-hidden",
        nargs="*",
        type=int,
        default=[],
        help=(
            "Optional fully connected hidden layers between the final pooled conv state "
            "and the output. Default is no FC hidden layer, matching the 2019 EP CNN."
        ),
    )

    parser.add_argument(
        "--kernel-size",
        type=int,
        default=5,
        help="Convolution kernel size",
    )

    parser.add_argument(
        "--pool-size",
        type=int,
        default=2,
        help="Max-pooling kernel/stride",
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Convolution padding",
    )

    parser.add_argument(
        "--T1",
        type=int,
        default=100,
        help="Number of free-phase / settling Euler steps",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Euler time step",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Top-down feedback strength",
    )

    parser.add_argument(
        "--exploration",
        type=float,
        default=0.2,
        help="Fixed Max-Boltzmann exploration probability",
    )

    parser.add_argument(
        "--act",
        type=str,
        default="hard_sigmoid",
        choices=["relu", "leaky_relu", "hard_sigmoid", "tanh", "sigmoid"],
        help="Activation for convolutional and optional FC hidden states",
    )

    parser.add_argument(
        "--act-clamp",
        type=float,
        default=None,
        help=(
            "Optional bound applied after the hidden activation. "
            "For example, --act leaky_relu --act-clamp 1.2 gives "
            "[-1.2, 1.2], while ReLU is effectively bounded to [0, 1.2]."
        ),
    )

    parser.add_argument(
        "--output-act",
        type=str,
        default="hard_sigmoid",
        choices=["relu", "leaky_relu", "hard_sigmoid", "tanh", "sigmoid", "linear"],
        help="Output-state activation",
    )

    parser.add_argument(
        "--output-clamp",
        type=float,
        default=None,
        help="Optional symmetric clamp applied after the output activation",
    )

    parser.add_argument(
        "--lrs",
        nargs="+",
        type=float,
        default=[0.01],
        help=(
            "Learning rate(s), in BOTTOM-UP order: conv1, conv2, ..., "
            "FC bridge, optional FC hidden-to-hidden layers, output layer. "
            "Provide one value to share across all trainable layers, or one value per layer."
        ),
    )

    parser.add_argument(
        "--init-mode",
        type=str,
        default="pytorch",
        choices=["pytorch", "normal"],
        help=(
            "Weight initialization. 'pytorch' keeps the native nn.Conv2d/nn.Linear "
            "fan-in-scaled initialization (recommended for CNNs and closest to the "
            "reference NeurIPS-2019 EP CNN implementation). 'normal' uses N(0, init_scale^2)."
        ),
    )

    parser.add_argument(
        "--init-scale",
        type=float,
        default=0.05,
        help="Std. dev. used only when --init-mode normal",
    )

    parser.add_argument(
        "--mbs",
        type=int,
        default=64,
        help="Mini-batch size",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 on clusters unless you explicitly want more.",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save accuracy/time arrays and configuration after every epoch",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default="./results_nel_cnn",
        help="Root directory for saved results",
    )

    return parser


# ============================================================
# Utilities
# ============================================================


def hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return (1.0 + F.hardtanh(2.0 * x - 1.0)) * 0.5


def apply_activation(
    x: torch.Tensor,
    name: str,
    clamp_value: Optional[float] = None,
) -> torch.Tensor:
    if name == "relu":
        y = F.relu(x)
    elif name == "leaky_relu":
        y = F.leaky_relu(x)
    elif name == "hard_sigmoid":
        y = hard_sigmoid(x)
    elif name == "tanh":
        y = torch.tanh(x)
    elif name == "sigmoid":
        y = torch.sigmoid(x)
    elif name == "linear":
        y = x
    else:
        raise ValueError(f"Unknown activation: {name}")

    if clamp_value is not None:
        if clamp_value <= 0.0:
            raise ValueError("Clamp value must be > 0")
        y = torch.clamp(y, -clamp_value, clamp_value)

    return y


@dataclass
class DatasetSpec:
    in_channels: int
    image_size: int
    num_classes: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]


def get_dataset_spec(task: str) -> DatasetSpec:
    if task == "MNIST":
        return DatasetSpec(
            in_channels=1,
            image_size=28,
            num_classes=10,
            mean=(0.1307,),
            std=(0.3081,),
        )

    if task == "FMNIST":
        return DatasetSpec(
            in_channels=1,
            image_size=28,
            num_classes=10,
            mean=(0.2860,),
            std=(0.3530,),
        )

    if task == "CIFAR10":
        # Common dataset-level CIFAR-10 normalization.
        return DatasetSpec(
            in_channels=3,
            image_size=32,
            num_classes=10,
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        )

    raise ValueError(f"Unsupported task: {task}")


def build_datasets(task: str):
    spec = get_dataset_spec(task)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(spec.mean, spec.std),
    ])

    if task == "MNIST":
        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

    elif task == "FMNIST":
        train_dataset = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )

    elif task == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

    else:
        raise ValueError(f"Unsupported task: {task}")

    return train_dataset, test_dataset, spec


# ============================================================
# CNN-NEL model
# ============================================================


class NeuromodulatedEquilibriumCNN(nn.Module):
    """
    Reciprocal equilibrium CNN trained by single-equilibrium NEL.

    State order is BOTTOM-UP:
        conv1 pooled state,
        conv2 pooled state,
        ...,
        optional FC hidden states,
        output state.

    Weight order is also BOTTOM-UP:
        conv1, conv2, ..., FC bridge, ..., output FC.
    """

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        num_classes: int,
        channels: Sequence[int],
        fc_hidden: Sequence[int],
        kernel_size: int,
        pool_size: int,
        padding: int,
        T1: int,
        dt: float,
        gamma: float,
        exploration: float,
        hidden_act: str,
        hidden_clamp: Optional[float],
        output_act: str,
        output_clamp: Optional[float],
        learning_rates: Sequence[float],
        init_mode: str,
        init_scale: float,
        device: torch.device,
    ):
        super().__init__()

        if len(channels) < 1:
            raise ValueError("At least one convolutional layer is required")
        if any(c <= 0 for c in channels):
            raise ValueError("All channel sizes must be > 0")
        if any(h <= 0 for h in fc_hidden):
            raise ValueError("All FC hidden sizes must be > 0")
        if kernel_size <= 0 or pool_size <= 0:
            raise ValueError("kernel_size and pool_size must be > 0")
        if T1 <= 0 or dt <= 0.0:
            raise ValueError("T1 and dt must be > 0")
        if not (0.0 <= exploration <= 1.0):
            raise ValueError("exploration must be in [0, 1]")
        if init_mode not in ("pytorch", "normal"):
            raise ValueError("init_mode must be 'pytorch' or 'normal'")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be > 0")

        self.in_channels = in_channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.channels = list(channels)
        self.fc_hidden = list(fc_hidden)
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.padding = padding
        self.T1 = T1
        self.dt = dt
        self.gamma = gamma
        self.exploration = exploration
        self.hidden_act = hidden_act
        self.hidden_clamp = hidden_clamp
        self.output_act = output_act
        self.output_clamp = output_clamp
        self.init_mode = init_mode
        self.init_scale = init_scale
        self.device = device

        # --------------------------------------------------------
        # Convolutional weights, bottom-up order.
        # --------------------------------------------------------
        conv_layers = []
        c_in = in_channels
        for c_out in self.channels:
            layer = nn.Conv2d(
                c_in,
                c_out,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding,
                bias=True,
            ).to(device)
            conv_layers.append(layer)
            c_in = c_out

        self.conv = nn.ModuleList(conv_layers)

        # IMPORTANT:
        # Do not force the MLP N(0, 0.05^2) initialization onto CNN layers.
        # The reciprocal convolutional dynamics can become unstable because
        # fan-in differs strongly across conv/FC layers. By default we keep
        # PyTorch's native fan-in-scaled Conv2d initialization, as in the
        # attached NeurIPS-2019 CNN implementation.
        if self.init_mode == "normal":
            with torch.no_grad():
                for layer in self.conv:
                    layer.weight.normal_(mean=0.0, std=self.init_scale)
                    if layer.bias is not None:
                        layer.bias.zero_()

        # Infer pooled and pre-pool tensor shapes exactly.
        (
            self.conv_state_shapes,
            self.conv_prepool_shapes,
        ) = self._infer_conv_shapes()

        final_conv_shape = self.conv_state_shapes[-1]
        final_conv_features = int(np.prod(final_conv_shape))

        # --------------------------------------------------------
        # FC layers, bottom-up order.
        # First FC is the bridge from flattened final conv state.
        # If fc_hidden is empty, it maps directly to the output.
        # --------------------------------------------------------
        fc_sizes = [final_conv_features] + self.fc_hidden + [num_classes]
        fc_layers = []

        for pre_size, post_size in zip(fc_sizes[:-1], fc_sizes[1:]):
            layer = nn.Linear(pre_size, post_size, bias=True).to(device)
            fc_layers.append(layer)

        self.fc = nn.ModuleList(fc_layers)

        if self.init_mode == "normal":
            with torch.no_grad():
                for layer in self.fc:
                    layer.weight.normal_(mean=0.0, std=self.init_scale)
                    if layer.bias is not None:
                        layer.bias.zero_()

        self.num_conv = len(self.conv)
        self.num_fc = len(self.fc)
        self.num_fc_hidden = len(self.fc_hidden)
        self.num_weight_layers = self.num_conv + self.num_fc
        self.num_states = self.num_conv + self.num_fc_hidden + 1

        if len(learning_rates) == 1:
            self.lrs = list(learning_rates) * self.num_weight_layers
        elif len(learning_rates) == self.num_weight_layers:
            self.lrs = list(learning_rates)
        else:
            raise ValueError(
                f"--lrs must contain either 1 value or {self.num_weight_layers} values "
                f"for {self.num_conv} conv + {self.num_fc} FC trainable layers"
            )

    # ------------------------------------------------------------
    # Shape inference
    # ------------------------------------------------------------

    @torch.no_grad()
    def _infer_conv_shapes(self):
        x = torch.zeros(
            1,
            self.in_channels,
            self.image_size,
            self.image_size,
            device=self.device,
        )

        pooled_shapes = []
        prepool_shapes = []

        for layer in self.conv:
            prepool = layer(x)
            if prepool.size(-1) < self.pool_size or prepool.size(-2) < self.pool_size:
                raise ValueError(
                    "Convolution/pooling architecture collapses spatial dimensions. "
                    "Change --channels/--kernel-size/--pool-size/--padding."
                )

            pooled = F.max_pool2d(
                prepool,
                kernel_size=self.pool_size,
                stride=self.pool_size,
            )

            prepool_shapes.append(tuple(prepool.shape[1:]))
            pooled_shapes.append(tuple(pooled.shape[1:]))
            x = pooled

        return pooled_shapes, prepool_shapes

    # ------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------

    @torch.no_grad()
    def init_states(self, batch_size: int):
        states = []

        for shape in self.conv_state_shapes:
            states.append(
                torch.zeros(
                    (batch_size,) + tuple(shape),
                    device=self.device,
                )
            )

        for hidden_size in self.fc_hidden:
            states.append(
                torch.zeros(batch_size, hidden_size, device=self.device)
            )

        states.append(
            torch.zeros(batch_size, self.num_classes, device=self.device)
        )

        return states

    # ------------------------------------------------------------
    # Pool metadata
    # ------------------------------------------------------------

    @torch.no_grad()
    def _pool_bottom_up(self, x: torch.Tensor, conv_states: Sequence[torch.Tensor]):
        """
        Compute bottom-up pooled drives and max-pool switches from a fixed set
        of lower activities. The returned switches are also used to transpose
        the pooling operation in top-down dynamics and in the NEL update.
        """
        pooled_drives = []
        pool_indices = []
        prepool_sizes = []

        lower = x

        for i, layer in enumerate(self.conv):
            if i > 0:
                lower = conv_states[i - 1]

            prepool = layer(lower)
            pooled, indices = F.max_pool2d(
                prepool,
                kernel_size=self.pool_size,
                stride=self.pool_size,
                return_indices=True,
            )

            pooled_drives.append(pooled)
            pool_indices.append(indices)
            prepool_sizes.append(tuple(prepool.shape))

        return pooled_drives, pool_indices, prepool_sizes

    # ------------------------------------------------------------
    # Transpose helpers
    # ------------------------------------------------------------

    @torch.no_grad()
    def _unpool(
        self,
        value: torch.Tensor,
        indices: torch.Tensor,
        output_size: Sequence[int],
    ) -> torch.Tensor:
        return F.max_unpool2d(
            value,
            indices,
            kernel_size=self.pool_size,
            stride=self.pool_size,
            output_size=output_size,
        )

    @torch.no_grad()
    def _transpose_conv_pool(
        self,
        upper_value: torch.Tensor,
        upper_conv_index: int,
        pool_indices: Sequence[torch.Tensor],
        prepool_sizes: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """
        Transpose of:
            lower --Conv_i--> prepool --MaxPool--> upper

        Uses max-pool switches, then ConvTranspose with the tied forward kernel.
        No activation derivative is used.
        """
        unpooled = self._unpool(
            upper_value,
            pool_indices[upper_conv_index],
            prepool_sizes[upper_conv_index],
        )

        return F.conv_transpose2d(
            unpooled,
            weight=self.conv[upper_conv_index].weight,
            bias=None,
            stride=1,
            padding=self.padding,
        )

    # ------------------------------------------------------------
    # Free-equilibrium relaxation dynamics
    # ------------------------------------------------------------

    @torch.no_grad()
    def free_phase(self, x: torch.Tensor):
        batch_size = x.size(0)
        states = self.init_states(batch_size)

        for _ in range(self.T1):
            old_states = states
            old_conv = old_states[: self.num_conv]

            pooled_drives, pool_indices, prepool_sizes = self._pool_bottom_up(
                x,
                old_conv,
            )

            new_states = []

            # ====================================================
            # Convolutional states
            # ====================================================
            for i in range(self.num_conv):
                bottom_up = pooled_drives[i]

                if i < self.num_conv - 1:
                    upper_old = old_conv[i + 1]
                    top_down = self._transpose_conv_pool(
                        upper_old,
                        upper_conv_index=i + 1,
                        pool_indices=pool_indices,
                        prepool_sizes=prepool_sizes,
                    )
                else:
                    # Top-down from first FC state. If there is no FC hidden
                    # layer, this is directly the output state.
                    upper_old = old_states[self.num_conv]
                    top_down_flat = upper_old @ self.fc[0].weight
                    top_down = top_down_flat.view_as(old_conv[i])

                drive = bottom_up + self.gamma * top_down
                target_activity = apply_activation(
                    drive,
                    self.hidden_act,
                    self.hidden_clamp,
                )

                old = old_conv[i]
                new = old + self.dt * (-old + target_activity)
                new_states.append(new)

            # ====================================================
            # Optional fully connected hidden states
            # ====================================================
            for j in range(self.num_fc_hidden):
                state_index = self.num_conv + j
                old = old_states[state_index]

                if j == 0:
                    lower = old_conv[-1].reshape(batch_size, -1)
                else:
                    lower = old_states[state_index - 1]

                bottom_up = self.fc[j](lower)
                upper_old = old_states[state_index + 1]
                top_down = upper_old @ self.fc[j + 1].weight

                drive = bottom_up + self.gamma * top_down
                target_activity = apply_activation(
                    drive,
                    self.hidden_act,
                    self.hidden_clamp,
                )

                new = old + self.dt * (-old + target_activity)
                new_states.append(new)

            # ====================================================
            # Output state
            # ====================================================
            output_old = old_states[-1]

            if self.num_fc_hidden > 0:
                lower = old_states[-2]
            else:
                lower = old_conv[-1].reshape(batch_size, -1)

            output_drive = self.fc[-1](lower)
            output_target = apply_activation(
                output_drive,
                self.output_act,
                self.output_clamp,
            )

            output_new = output_old + self.dt * (-output_old + output_target)
            new_states.append(output_new)

            states = new_states

        # Recompute pool switches from the final lower states so the learning
        # rule uses metadata consistent with the final free-phase activities.
        final_conv = states[: self.num_conv]
        _, final_indices, final_prepool_sizes = self._pool_bottom_up(
            x,
            final_conv,
        )

        return states, final_indices, final_prepool_sizes

    # ------------------------------------------------------------
    # Max-Boltzmann selection
    # ------------------------------------------------------------

    @torch.no_grad()
    def select_class(self, output: torch.Tensor):
        if not torch.isfinite(output).all():
            finite_fraction = torch.isfinite(output).float().mean().item()
            raise RuntimeError(
                "Non-finite output encountered before class selection "
                f"(finite fraction={finite_fraction:.6f}). "
                "The neural dynamics or weight updates have become unstable."
            )

        pred = torch.argmax(output, dim=1)
        selected = pred.clone()

        if self.exploration > 0.0:
            explore_mask = (
                torch.rand(output.size(0), device=self.device) < self.exploration
            )

            sampled = torch.distributions.Categorical(logits=output).sample()
            selected[explore_mask] = sampled[explore_mask]

        return pred, selected

    # ------------------------------------------------------------
    # Binary reward and output reward-gradient signal
    # ------------------------------------------------------------

    @torch.no_grad()
    def compute_reward(
        self,
        selected: torch.Tensor,
        labels: torch.Tensor,
        output: torch.Tensor,
    ):
        return (selected == labels).to(dtype=output.dtype)

    @torch.no_grad()
    def reward_gradient_output(
        self,
        output: torch.Tensor,
        selected: torch.Tensor,
        reward: torch.Tensor,
    ):
        batch_size = output.size(0)
        idx = torch.arange(batch_size, device=self.device)

        selected_output = output[idx, selected]
        selected_signal = reward - selected_output

        reward_out = torch.zeros_like(output)
        reward_out[idx, selected] = selected_signal

        return reward_out

    # ------------------------------------------------------------
    # NEL update
    # ------------------------------------------------------------

    @torch.no_grad()
    def update_weights(
        self,
        x: torch.Tensor,
        states: Sequence[torch.Tensor],
        pool_indices: Sequence[torch.Tensor],
        prepool_sizes: Sequence[Sequence[int]],
        selected: torch.Tensor,
        reward: torch.Tensor,
    ):
        batch_size = x.size(0)
        conv_states = list(states[: self.num_conv])
        fc_states = list(states[self.num_conv :])
        output = states[-1]

        # ========================================================
        # 1) Build modulators, starting at output
        # ========================================================
        output_mod = self.reward_gradient_output(
            output,
            selected,
            reward,
        )

        # FC modulators are ordered bottom-up over postsynaptic FC states:
        #   fc_mod[0] -> first FC state (or output if no FC hidden)
        #   ...
        #   fc_mod[-1] -> output
        fc_mod = [None for _ in range(self.num_fc)]
        fc_mod[-1] = output_mod

        for j in range(self.num_fc - 2, -1, -1):
            fc_mod[j] = fc_mod[j + 1] @ self.fc[j + 1].weight

        # Modulator for the final conv state comes through the first FC bridge.
        last_conv_mod_flat = fc_mod[0] @ self.fc[0].weight
        conv_mod = [None for _ in range(self.num_conv)]
        conv_mod[-1] = last_conv_mod_flat.view_as(conv_states[-1])

        # Propagate through conv+pool operators in reverse.
        for i in range(self.num_conv - 2, -1, -1):
            conv_mod[i] = self._transpose_conv_pool(
                conv_mod[i + 1],
                upper_conv_index=i + 1,
                pool_indices=pool_indices,
                prepool_sizes=prepool_sizes,
            )

        # ========================================================
        # 2) Compute ALL updates before changing ANY parameter
        # ========================================================
        conv_dW = []
        conv_db = []
        fc_dW = []
        fc_db = []

        # --------------------------------------------------------
        # Convolutional local updates
        # --------------------------------------------------------
        for i in range(self.num_conv):
            pre = x if i == 0 else conv_states[i - 1]
            post = conv_states[i]
            modulation = conv_mod[i]

            # Same NEL post-factor as in the MLP implementation.
            post_factor_pool = modulation * post

            # Transpose the max-pool routing to the pre-pool conv map.
            post_factor_prepool = self._unpool(
                post_factor_pool,
                pool_indices[i],
                prepool_sizes[i],
            )

            layer = self.conv[i]

            # Correlation between presynaptic activity and the unpooled
            # postsynaptic NEL factor. This is the convolutional analogue of:
            #     (post_factor.T @ pre) / batch_size
            layer_dW = conv2d_weight(
                pre,
                layer.weight.shape,
                post_factor_prepool,
                stride=1,
                padding=self.padding,
                dilation=1,
                groups=1,
            ) / batch_size

            layer_db = post_factor_prepool.sum(dim=(0, 2, 3)) / batch_size

            conv_dW.append(layer_dW)
            conv_db.append(layer_db)

        # --------------------------------------------------------
        # Fully connected local updates
        # --------------------------------------------------------
        # Build FC activities in bottom-up order:
        #   pre for fc[0] is flattened final conv state,
        #   post is first FC hidden or output.
        fc_activity = []
        fc_activity.append(conv_states[-1].reshape(batch_size, -1))
        fc_activity.extend(fc_states)

        for j in range(self.num_fc):
            pre = fc_activity[j]
            post = fc_activity[j + 1]
            modulation = fc_mod[j]

            post_factor = modulation * post
            layer_dW = (post_factor.T @ pre) / batch_size
            layer_db = post_factor.mean(dim=0)

            fc_dW.append(layer_dW)
            fc_db.append(layer_db)

        # ========================================================
        # 3) Apply parameter updates, bottom-up LR order
        # ========================================================
        lr_index = 0

        for i in range(self.num_conv):
            self.conv[i].weight += self.lrs[lr_index] * conv_dW[i]
            self.conv[i].bias += self.lrs[lr_index] * conv_db[i]
            lr_index += 1

        for j in range(self.num_fc):
            self.fc[j].weight += self.lrs[lr_index] * fc_dW[j]
            self.fc[j].bias += self.lrs[lr_index] * fc_db[j]
            lr_index += 1

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        states, _, _ = self.free_phase(x)
        output = states[-1]
        return torch.argmax(output, dim=1), output

    # ------------------------------------------------------------
    # Human-readable architecture summary
    # ------------------------------------------------------------

    def architecture_summary(self):
        lines = []
        lines.append(
            f"Input          : {self.in_channels} x {self.image_size} x {self.image_size}"
        )

        c_in = self.in_channels
        for i, (c_out, pooled_shape, prepool_shape) in enumerate(
            zip(self.channels, self.conv_state_shapes, self.conv_prepool_shapes),
            start=1,
        ):
            lines.append(
                f"Conv{i:<2}         : {c_in} -> {c_out}, "
                f"pre-pool {prepool_shape[1]}x{prepool_shape[2]}, "
                f"pooled {pooled_shape[1]}x{pooled_shape[2]}"
            )
            c_in = c_out

        if self.fc_hidden:
            for i, size in enumerate(self.fc_hidden, start=1):
                lines.append(f"FC hidden {i:<2}   : {size}")

        lines.append(f"Output         : {self.num_classes}")
        return "\n".join(lines)


# ============================================================
# Evaluation helper
# ============================================================


@torch.no_grad()
def evaluate(model, loader, device):
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        pred, _ = model.predict(images)

        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return correct / total


# ============================================================
# Saving helpers
# ============================================================


def build_run_name(args, model):
    channels_str = "-".join(map(str, args.channels))
    fc_str = "none" if len(args.fc_hidden) == 0 else "-".join(map(str, args.fc_hidden))
    lr_str = "-".join(str(x) for x in model.lrs)
    clamp_str = "none" if args.act_clamp is None else str(args.act_clamp)

    return (
        f"NELCNN_{args.task}"
        f"_ch{channels_str}"
        f"_fc{fc_str}"
        f"_k{args.kernel_size}"
        f"_p{args.pool_size}"
        f"_pad{args.padding}"
        f"_T1{args.T1}"
        f"_dt{args.dt}"
        f"_gamma{args.gamma}"
        f"_exp{args.exploration}"
        f"_act{args.act}"
        f"_clamp{clamp_str}"
        f"_out{args.output_act}"
        f"_lr{lr_str}"
        f"_mbs{args.mbs}"
        f"_seed{args.seed}"
    )


def save_results(
    save_path,
    online_train_acc,
    test_acc,
    epoch_time,
    config,
):
    """Save histories and the exact run configuration.

    ``online_train_acc`` is accumulated from free-equilibrium predictions
    during training. ``test_acc`` is evaluated after each completed epoch.
    The historical filename ``train_acc.npy`` is retained for compatibility
    with the experiment-analysis scripts.
    """
    os.makedirs(save_path, exist_ok=True)

    train_acc_array = np.asarray(online_train_acc, dtype=np.float64)
    test_acc_array = np.asarray(test_acc, dtype=np.float64)
    epoch_time_array = np.asarray(epoch_time, dtype=np.float64)

    np.save(os.path.join(save_path, "train_acc.npy"), train_acc_array)
    np.save(os.path.join(save_path, "test_acc.npy"), test_acc_array)
    np.save(os.path.join(save_path, "epoch_time.npy"), epoch_time_array)

    np.save(
        os.path.join(save_path, "train_test_acc.npy"),
        np.stack((train_acc_array, test_acc_array), axis=0),
    )

    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ============================================================
# Main
# ============================================================


def main():
    parser = build_parser()
    args = parser.parse_args()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    if args.T1 <= 0:
        raise ValueError("--T1 must be > 0")
    if args.dt <= 0.0:
        raise ValueError("--dt must be > 0")
    if args.mbs <= 0:
        raise ValueError("--mbs must be > 0")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if not (0.0 <= args.exploration <= 1.0):
        raise ValueError("--exploration must be in [0, 1]")
    if args.kernel_size <= 0 or args.pool_size <= 0:
        raise ValueError("--kernel-size and --pool-size must be > 0")
    if args.padding < 0:
        raise ValueError("--padding must be >= 0")
    if args.gamma < 0.0:
        raise ValueError("--gamma must be >= 0")
    if args.act_clamp is not None and args.act_clamp <= 0.0:
        raise ValueError("--act-clamp must be > 0 when provided")
    if args.output_clamp is not None and args.output_clamp <= 0.0:
        raise ValueError("--output-clamp must be > 0 when provided")

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    train_dataset, test_dataset, spec = build_datasets(args.task)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.mbs,
        shuffle=True,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.mbs,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = NeuromodulatedEquilibriumCNN(
        in_channels=spec.in_channels,
        image_size=spec.image_size,
        num_classes=spec.num_classes,
        channels=args.channels,
        fc_hidden=args.fc_hidden,
        kernel_size=args.kernel_size,
        pool_size=args.pool_size,
        padding=args.padding,
        T1=args.T1,
        dt=args.dt,
        gamma=args.gamma,
        exploration=args.exploration,
        hidden_act=args.act,
        hidden_clamp=args.act_clamp,
        output_act=args.output_act,
        output_clamp=args.output_clamp,
        learning_rates=args.lrs,
        init_mode=args.init_mode,
        init_scale=args.init_scale,
        device=device,
    )

    run_name = build_run_name(args, model)
    save_path = os.path.join(args.save_path, run_name)

    config = vars(args).copy()
    config["device"] = str(device)
    config["resolved_lrs"] = model.lrs
    config["conv_state_shapes"] = [list(x) for x in model.conv_state_shapes]
    config["conv_prepool_shapes"] = [list(x) for x in model.conv_prepool_shapes]

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------
    print("=" * 84)
    print("Neuromodulated Equilibrium Learning (NEL) - Convolutional Network")
    print("=" * 84)
    print(f"Task              : {args.task}")
    print(f"Channels          : {args.channels}")
    print(f"FC hidden         : {args.fc_hidden if args.fc_hidden else 'None'}")
    print(f"Kernel / Pool     : {args.kernel_size} / {args.pool_size}")
    print(f"Padding           : {args.padding}")
    print(f"T1                : {args.T1}")
    print(f"dt                : {args.dt}")
    print(f"gamma             : {args.gamma}")
    print(f"Exploration       : {args.exploration}")
    print(f"Hidden act        : {args.act}")
    print(f"Hidden clamp      : {args.act_clamp}")
    print(f"Output act        : {args.output_act}")
    print(f"Output clamp      : {args.output_clamp}")
    print(f"Learning rates    : {model.lrs}")
    print(f"Initialization    : {args.init_mode}" + (
        f" (std={args.init_scale})" if args.init_mode == "normal" else ""
    ))
    print(f"LR order          : conv1 -> conv2 -> ... -> FC bridge -> ... -> output")
    print(f"Batch size        : {args.mbs}")
    print(f"Epochs            : {args.epochs}")
    print(f"Seed              : {args.seed}")
    print(f"Device            : {device}")
    print(f"Save              : {args.save}")
    if args.save:
        print(f"Save path         : {save_path}")
    print("-" * 84)
    print(model.architecture_summary())
    print("=" * 84)

    # --------------------------------------------------------
    # Accuracy/time histories
    # --------------------------------------------------------
    chance_accuracy = 100.0 / spec.num_classes
    online_train_acc = [chance_accuracy]
    test_acc = [chance_accuracy]
    epoch_time = []

    if args.save:
        os.makedirs(save_path, exist_ok=True)
        save_results(save_path, online_train_acc, test_acc, epoch_time, config)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    total_start = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()

        run_correct = 0
        run_selected_correct = 0
        run_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 1. Free phase only
            states, pool_indices, prepool_sizes = model.free_phase(images)
            output = states[-1]

            # 2. Max-Boltzmann class selection
            pred, selected = model.select_class(output)

            # 3. Binary reward
            reward = model.compute_reward(selected, labels, output)

            # 4. Single-equilibrium NEL update
            model.update_weights(
                images,
                states,
                pool_indices,
                prepool_sizes,
                selected,
                reward,
            )

            # Online training statistics from the pre-update free-equilibrium output.
            run_correct += (pred == labels).sum().item()
            run_selected_correct += reward.sum().item()
            run_total += labels.size(0)

        run_acc = run_correct / run_total
        selected_reward_rate = run_selected_correct / run_total
        test_acc_t = evaluate(model, test_loader, device)

        elapsed = time.perf_counter() - epoch_start

        online_train_acc.append(100.0 * run_acc)
        test_acc.append(100.0 * test_acc_t)
        epoch_time.append(elapsed)

        print(
            "Epoch {:03d} | Train Acc: {:.2f}% | Test Acc: {:.2f}% | "
            "Selected Reward Rate: {:.3f} | Exploration: {:.2f} | Time: {:.1f}s".format(
                epoch + 1,
                100.0 * run_acc,
                100.0 * test_acc_t,
                selected_reward_rate,
                args.exploration,
                elapsed,
            )
        )

        if args.save:
            save_results(save_path, online_train_acc, test_acc, epoch_time, config)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------
    total_elapsed = time.perf_counter() - total_start
    test_acc_array = np.asarray(test_acc, dtype=np.float64)
    best_test_index = int(np.argmax(test_acc_array))

    print()
    print(
        "Best test accuracy: {:.2f}% at epoch {}".format(
            test_acc_array[best_test_index],
            best_test_index,
        )
    )
    print(f"Total training time: {total_elapsed / 60.0:.2f} min")

    if args.save:
        config["total_training_time_seconds"] = total_elapsed
        save_results(save_path, online_train_acc, test_acc, epoch_time, config)
        print()
        print("Saved files:")
        print(os.path.join(save_path, "train_acc.npy"))
        print(os.path.join(save_path, "test_acc.npy"))
        print(os.path.join(save_path, "train_test_acc.npy"))
        print(os.path.join(save_path, "epoch_time.npy"))
        print(os.path.join(save_path, "config.json"))


if __name__ == "__main__":
    main()
