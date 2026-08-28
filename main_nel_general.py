import argparse
import os
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# Neuromodulated Equilibrium Learning (NEL)
# General MLP implementation for arbitrary hidden-layer depth
# ============================================================


# ============================================================
# Command-line arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Neuromodulated Equilibrium Learning (NEL)"
)

parser.add_argument(
    "--task",
    type=str,
    default="MNIST",
    choices=["MNIST", "FMNIST"],
)

parser.add_argument(
    "--archi",
    nargs="+",
    type=int,
    default=[784, 512, 10],
    help=(
        "Network architecture including input and output sizes. "
        "Examples: --archi 784 512 10 or --archi 784 512 256 10"
    ),
)

parser.add_argument(
    "--T1",
    type=int,
    default=80,
    help="Number of free-phase Euler steps",
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
    help="Top-down feedback strength in the free-phase dynamics",
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
    default="relu",
    choices=[
        "relu",
        "leaky_relu",
        "hard_sigmoid",
        "tanh",
        "sigmoid",
    ],
    help="Activation for all hidden layers",
)

parser.add_argument(
    "--output-act",
    type=str,
    default="hard_sigmoid",
    choices=[
        "relu",
        "leaky_relu",
        "hard_sigmoid",
        "tanh",
        "sigmoid",
        "linear",
    ],
    help="Output-layer activation",
)

parser.add_argument(
    "--lrs",
    nargs="+",
    type=float,
    default=[0.1],
    help=(
        "Learning rate(s). "
        "Provide one value to share across all layers, "
        "or one value per weight layer. "
        "Examples: --lrs 0.1 or --lrs 0.1 0.1 0.1"
    ),
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
    "--save",
    action="store_true",
    help="Save accuracy arrays after each epoch",
)

parser.add_argument(
    "--save-path",
    type=str,
    default="./results_nel",
    help="Root directory for saved results",
)

args = parser.parse_args()


# ============================================================
# Validation
# ============================================================

if len(args.archi) < 3:
    raise ValueError(
        "--archi requires at least INPUT HIDDEN OUTPUT"
    )

if not (0.0 <= args.exploration <= 1.0):
    raise ValueError(
        "--exploration must be in [0, 1]"
    )

if args.T1 <= 0:
    raise ValueError(
        "--T1 must be > 0"
    )

if args.dt <= 0.0:
    raise ValueError(
        "--dt must be > 0"
    )

if args.mbs <= 0:
    raise ValueError(
        "--mbs must be > 0"
    )

if args.epochs <= 0:
    raise ValueError(
        "--epochs must be > 0"
    )


NUM_WEIGHT_LAYERS = len(args.archi) - 1
NUM_HIDDEN_LAYERS = len(args.archi) - 2


# ------------------------------------------------------------
# Learning-rate handling
#
# --lrs 0.1
#     -> same LR for every trainable layer
#
# --lrs 0.1 0.1 0.1
#     -> one LR for each weight layer
# ------------------------------------------------------------

if len(args.lrs) == 1:
    LRS = args.lrs * NUM_WEIGHT_LAYERS

elif len(args.lrs) == NUM_WEIGHT_LAYERS:
    LRS = args.lrs

else:
    raise ValueError(
        f"--lrs must contain either 1 value or "
        f"{NUM_WEIGHT_LAYERS} values for architecture {args.archi}"
    )


T1 = args.T1
DT = args.dt
GAMMA = args.gamma
EXPLORATION_RATE = args.exploration
BATCH_SIZE = args.mbs
EPOCHS = args.epochs
SEED = args.seed

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Activations
# ============================================================

def hard_sigmoid(x):
    return (
        1.0
        + F.hardtanh(2.0 * x - 1.0)
    ) * 0.5


def apply_activation(x, name):

    if name == "relu":
        #return F.relu(x)
        return torch.clamp(F.relu(x), 0.0, 1.0)

    if name == "leaky_relu":
        #return F.leaky_relu(x)
        #return torch.clamp(F.leaky_relu(x), -1.0, 1.0) #85.98%
        return torch.clamp(F.leaky_relu(x), -1.2, 1.2) #86.64%
        #return torch.clamp(F.leaky_relu(x), -2.0, 2.0) #85.99%

    if name == "hard_sigmoid":
        return hard_sigmoid(x)

    if name == "tanh":
        return torch.tanh(x)

    if name == "sigmoid":
        return torch.sigmoid(x)

    if name == "linear":
        return x

    raise ValueError(
        f"Unknown activation: {name}"
    )


# ============================================================
# Data
# ============================================================

if args.task == "MNIST":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,),
            (0.3081,),
        ),
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

elif args.task == "FMNIST":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.2860,),
            (0.3530,),
        ),
    ])

    train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

else:
    raise ValueError(
        f"Unsupported task: {args.task}"
    )


# Validate flattened input size
sample_image, _ = train_dataset[0]
dataset_input_size = int(sample_image.numel())

if args.archi[0] != dataset_input_size:
    raise ValueError(
        f"Architecture input size is {args.archi[0]}, "
        f"but {args.task} flattened input size is {dataset_input_size}"
    )


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# Model
# ============================================================

class NeuromodulatedEquilibriumLearning:

    def __init__(self, architecture):

        self.architecture = architecture
        self.num_weight_layers = len(architecture) - 1
        self.num_hidden_layers = len(architecture) - 2

        # W[k] connects layer k -> layer k+1.
        # Shape: [post_size, pre_size]
        self.W = []
        self.b = []

        for pre_size, post_size in zip(
            architecture[:-1],
            architecture[1:],
        ):

            weight = (
                0.05
                * torch.randn(
                    post_size,
                    pre_size,
                    device=DEVICE,
                )
            )

            bias = torch.zeros(
                post_size,
                device=DEVICE,
            )

            self.W.append(weight)
            self.b.append(bias)


    # ========================================================
    # Free phase
    #
    # Generalized discrete Euler dynamics.
    #
    # For hidden layer l:
    #
    #   x_l(t) =
    #       x_l(t-1)
    #       +
    #       dt[
    #          -x_l(t-1)
    #          +
    #          rho(
    #              W_l x_(l-1)(t-1)
    #              +
    #              gamma W_(l+1)^T x_(l+1)(t-1)
    #              +
    #              b_l
    #          )
    #       ]
    #
    # For the first hidden layer, x_(l-1) is the fixed input.
    #
    # For the output:
    #
    #   y(t) =
    #       y(t-1)
    #       +
    #       dt[
    #          -y(t-1)
    #          +
    #          rho(
    #              W_out x_last_hidden(t-1)
    #              +
    #              b_out
    #          )
    #       ]
    #
    # All layers use states from t-1.
    # ========================================================

    @torch.no_grad()
    def free_phase(self, x):

        batch_size = x.size(0)

        # states:
        # hidden1, hidden2, ..., output
        states = [
            torch.zeros(
                batch_size,
                layer_size,
                device=DEVICE,
            )
            for layer_size in self.architecture[1:]
        ]


        for _ in range(T1):

            old_states = states
            new_states = []


            # =================================================
            # Hidden layers
            # =================================================

            for hidden_idx in range(
                self.num_hidden_layers
            ):

                # --------------------------------------------
                # Lower-layer activity
                # --------------------------------------------

                if hidden_idx == 0:
                    lower_activity = x
                else:
                    lower_activity = old_states[
                        hidden_idx - 1
                    ]


                # --------------------------------------------
                # Bottom-up drive
                #
                # W[hidden_idx]:
                # lower -> current hidden
                # --------------------------------------------

                bottom_up = (
                    lower_activity
                    @ self.W[hidden_idx].T
                )


                # --------------------------------------------
                # Top-down drive
                #
                # old_states[hidden_idx + 1]
                # is the upper layer activity.
                #
                # W[hidden_idx + 1] maps:
                # current hidden -> upper layer.
                #
                # Therefore:
                #
                # upper @ W[hidden_idx + 1]
                #
                # is transpose feedback.
                # --------------------------------------------

                upper_activity = old_states[
                    hidden_idx + 1
                ]

                top_down = (
                    upper_activity
                    @ self.W[hidden_idx + 1]
                )


                hidden_drive = (
                    bottom_up
                    + GAMMA * top_down
                    + self.b[hidden_idx]
                )

                hidden_activation = apply_activation(
                    hidden_drive,
                    args.act,
                )

                hidden_old = old_states[
                    hidden_idx
                ]

                hidden_new = (
                    hidden_old
                    + DT * (
                        -hidden_old
                        + hidden_activation
                    )
                )

                new_states.append(
                    hidden_new
                )


            # =================================================
            # Output layer
            # =================================================

            last_hidden_old = old_states[
                self.num_hidden_layers - 1
            ]

            output_old = old_states[-1]

            output_drive = (
                last_hidden_old
                @ self.W[-1].T
                + self.b[-1]
            )

            output_activation = apply_activation(
                output_drive,
                args.output_act,
            )

            output_new = (
                output_old
                + DT * (
                    -output_old
                    + output_activation
                )
            )

            new_states.append(
                output_new
            )

            states = new_states


        hidden_states = states[:-1]
        output = states[-1]

        return hidden_states, output


    # ========================================================
    # Max-Boltzmann selection
    # ========================================================

    @torch.no_grad()
    def select_class(self, output):

        pred = torch.argmax(
            output,
            dim=1,
        )

        selected = pred.clone()


        if EXPLORATION_RATE > 0.0:

            explore_mask = (
                torch.rand(
                    output.size(0),
                    device=DEVICE,
                )
                < EXPLORATION_RATE
            )

            sampled = (
                torch.distributions.Categorical(
                    logits=output
                ).sample()
            )

            selected[
                explore_mask
            ] = sampled[
                explore_mask
            ]


        return pred, selected


    # ========================================================
    # Binary reward
    # ========================================================

    @torch.no_grad()
    def compute_reward(
        self,
        selected,
        labels,
        output,
    ):

        return (
            selected == labels
        ).to(
            dtype=output.dtype
        )


    # ========================================================
    # Reward-Based EP Eq. (18) / Eq. (19)
    #
    # C_a = 1/2 (r - y_a)^2
    #
    # -grad_y C_a = (r - y_a)e_a
    # ========================================================

    @torch.no_grad()
    def reward_gradient_output(
        self,
        output,
        selected,
        reward,
    ):

        batch_size = output.size(0)

        idx = torch.arange(
            batch_size,
            device=DEVICE,
        )

        selected_output = output[
            idx,
            selected,
        ]

        selected_signal = (
            reward
            - selected_output
        )

        reward_out = torch.zeros_like(
            output
        )

        reward_out[
            idx,
            selected,
        ] = selected_signal

        return reward_out


    # ========================================================
    # NEL update
    #
    # General arbitrary-depth version:
    #
    #   M_output = (r - y_a)e_a
    #
    #   M_l = M_(l+1) W_(l+1)
    #
    #   dW_l =
    #       (M_l * post_l)^T pre_l / batch
    #
    # No activation derivative.
    # No nudged phase.
    # No contrastive EP update.
    # ========================================================

    @torch.no_grad()
    def update_weights(
        self,
        x,
        hidden_states,
        output,
        selected,
        reward,
    ):

        batch_size = x.size(0)


        # ----------------------------------------------------
        # Activities:
        #
        # input, hidden1, hidden2, ..., output
        # ----------------------------------------------------

        activities = (
            [x]
            + hidden_states
            + [output]
        )


        # ----------------------------------------------------
        # modulators[k] corresponds to the postsynaptic
        # layer of W[k].
        #
        # Example 784-512-256-10:
        #
        # modulators[0] -> hidden1
        # modulators[1] -> hidden2
        # modulators[2] -> output
        # ----------------------------------------------------

        modulators = [
            None
            for _ in range(
                self.num_weight_layers
            )
        ]


        # ====================================================
        # Output modulator: Eq. (19)
        # ====================================================

        modulators[-1] = (
            self.reward_gradient_output(
                output,
                selected,
                reward,
            )
        )


        # ====================================================
        # Transpose feedback through all hidden layers
        #
        # No activation derivative.
        # ====================================================

        for k in range(
            self.num_weight_layers - 2,
            -1,
            -1,
        ):

            modulators[k] = (
                modulators[k + 1]
                @ self.W[k + 1]
            )


        # ====================================================
        # Compute all updates before modifying any weights
        # ====================================================

        dW = []
        db = []


        for k in range(
            self.num_weight_layers
        ):

            pre = activities[k]
            post = activities[k + 1]
            modulation = modulators[k]

            post_factor = (
                modulation
                * post
            )

            layer_dW = (
                post_factor.T
                @ pre
            ) / batch_size

            layer_db = (
                post_factor.mean(
                    dim=0
                )
            )

            dW.append(
                layer_dW
            )

            db.append(
                layer_db
            )


        # ====================================================
        # Apply updates
        # ====================================================

        for k in range(
            self.num_weight_layers
        ):

            self.W[k] += (
                LRS[k]
                * dW[k]
            )

            self.b[k] += (
                LRS[k]
                * db[k]
            )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
):

    correct = 0
    total = 0


    for images, labels in loader:

        x = images.view(
            images.size(0),
            -1,
        ).to(DEVICE)

        labels = labels.to(
            DEVICE
        )

        _, output = (
            model.free_phase(x)
        )

        pred = torch.argmax(
            output,
            dim=1,
        )

        correct += (
            pred == labels
        ).sum().item()

        total += (
            labels.size(0)
        )


    return (
        correct / total
    )


# ============================================================
# Run naming / saving
# ============================================================

lr_string = "-".join(
    str(lr)
    for lr in LRS
)

run_name = (
    f"NEL_{args.task}"
    f"_arch{'-'.join(map(str, args.archi))}"
    f"_T1{T1}"
    f"_dt{DT}"
    f"_gamma{GAMMA}"
    f"_exp{EXPLORATION_RATE}"
    f"_act{args.act}"
    f"_outact{args.output_act}"
    f"_lr{lr_string}"
    f"_mbs{BATCH_SIZE}"
    f"_seed{SEED}"
)

SAVE_PATH = os.path.join(
    args.save_path,
    run_name,
)


# ============================================================
# Print configuration
# ============================================================

print("=" * 76)
print("Neuromodulated Equilibrium Learning (NEL)")
print("=" * 76)
print(f"Task           : {args.task}")
print(f"Architecture   : {args.archi}")
print(f"Hidden layers  : {NUM_HIDDEN_LAYERS}")
print(f"T1             : {T1}")
print(f"dt             : {DT}")
print(f"gamma          : {GAMMA}")
print(f"Exploration    : {EXPLORATION_RATE}")
print(f"Hidden act     : {args.act}")
print(f"Output act     : {args.output_act}")
print(f"Learning rates : {LRS}")
print(f"Batch size     : {BATCH_SIZE}")
print(f"Epochs         : {EPOCHS}")
print(f"Seed           : {SEED}")
print(f"Device         : {DEVICE}")
print(f"Save           : {args.save}")

if args.save:
    print(f"Save path      : {SAVE_PATH}")

print("=" * 76)


# ============================================================
# Initialize model
# ============================================================

model = (
    NeuromodulatedEquilibriumLearning(
        args.archi
    )
)


# ============================================================
# Accuracy histories
# ============================================================

chance_accuracy = (
    100.0
    / args.archi[-1]
)

train_acc = [
    chance_accuracy
]

test_acc = [
    chance_accuracy
]


if args.save:

    os.makedirs(
        SAVE_PATH,
        exist_ok=True,
    )


# ============================================================
# Training
# ============================================================

for epoch in range(EPOCHS):

    run_correct = 0
    run_selected_correct = 0
    run_total = 0


    for images, labels in train_loader:

        x = images.view(
            images.size(0),
            -1,
        ).to(DEVICE)

        labels = labels.to(
            DEVICE
        )


        # 1. Free phase only
        hidden_states, output = (
            model.free_phase(x)
        )


        # 2. Max-Boltzmann class selection
        pred, selected = (
            model.select_class(
                output
            )
        )


        # 3. Binary reward
        reward = (
            model.compute_reward(
                selected,
                labels,
                output,
            )
        )


        # 4. NEL update
        model.update_weights(
            x,
            hidden_states,
            output,
            selected,
            reward,
        )


        # Statistics
        run_correct += (
            pred == labels
        ).sum().item()

        run_selected_correct += (
            reward.sum().item()
        )

        run_total += (
            labels.size(0)
        )


    run_acc = (
        run_correct
        / run_total
    )

    selected_reward_rate = (
        run_selected_correct
        / run_total
    )

    test_acc_t = evaluate(
        model,
        test_loader,
    )


    train_acc.append(
        100.0
        * run_acc
    )

    test_acc.append(
        100.0
        * test_acc_t
    )


    print(
        "Epoch {:02d} | "
        "Train Acc: {:.2f}% | "
        "Test Acc: {:.2f}% | "
        "Selected Reward Rate: {:.3f} | "
        "Exploration: {:.2f}".format(
            epoch + 1,
            100.0 * run_acc,
            100.0 * test_acc_t,
            selected_reward_rate,
            EXPLORATION_RATE,
        )
    )


    # ========================================================
    # Save NPY after every epoch
    # ========================================================

    if args.save:

        train_acc_array = np.asarray(
            train_acc,
            dtype=np.float64,
        )

        test_acc_array = np.asarray(
            test_acc,
            dtype=np.float64,
        )


        np.save(
            os.path.join(
                SAVE_PATH,
                "train_acc.npy",
            ),
            train_acc_array,
        )


        np.save(
            os.path.join(
                SAVE_PATH,
                "test_acc.npy",
            ),
            test_acc_array,
        )


        np.save(
            os.path.join(
                SAVE_PATH,
                "train_test_acc.npy",
            ),
            np.stack(
                (
                    train_acc_array,
                    test_acc_array,
                ),
                axis=0,
            ),
        )


# ============================================================
# Final summary
# ============================================================

test_acc_array = np.asarray(
    test_acc,
    dtype=np.float64,
)

best_test_index = int(
    np.argmax(
        test_acc_array
    )
)


print()

print(
    "Best test accuracy: "
    "{:.2f}% at epoch {}".format(
        test_acc_array[
            best_test_index
        ],
        best_test_index,
    )
)


if args.save:

    print()
    print("Saved accuracy files:")

    print(
        os.path.join(
            SAVE_PATH,
            "train_acc.npy",
        )
    )

    print(
        os.path.join(
            SAVE_PATH,
            "test_acc.npy",
        )
    )

    print(
        os.path.join(
            SAVE_PATH,
            "train_test_acc.npy",
        )
    )
