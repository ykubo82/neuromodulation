import argparse
import os
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# Direct Cost-Feedback Equilibrium Learning
# Controlled variant of NEL for testing direct propagation of -dC/dy
# General MLP implementation for arbitrary hidden-layer depth
# ============================================================


# ============================================================
# Command-line arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Direct Cost-Feedback Equilibrium Learning"
)


parser.add_argument(
    "--task",
    type=str,
    default="MNIST",
    choices=["MNIST", "FMNIST"],
    help="Dataset: MNIST or FMNIST",
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
    "--act-clamp",
    type=float,
    default=None,
    help=(
        "Optional symmetric clamp applied after the hidden activation. "
        "Example: --act leaky_relu --act-clamp 1.2 gives [-1.2, 1.2]."
    ),
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
    default="./results_direct_cost_feedback",
    help="Root directory for saved results",
)

args = parser.parse_args()

TASK = args.task


# ============================================================
# Validation
# ============================================================

if len(args.archi) < 3:
    raise ValueError(
        "--archi requires at least INPUT HIDDEN OUTPUT"
    )


if args.T1 <= 0:
    raise ValueError(
        "--T1 must be > 0"
    )

if args.dt <= 0.0:
    raise ValueError(
        "--dt must be > 0"
    )

if args.act_clamp is not None and args.act_clamp <= 0.0:
    raise ValueError(
        "--act-clamp must be > 0 when provided"
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


def apply_activation(x, name, clamp_value=None):

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
        raise ValueError(
            f"Unknown activation: {name}"
        )

    if clamp_value is not None:
        y = torch.clamp(
            y,
            -clamp_value,
            clamp_value,
        )

    return y


# ============================================================
# Data: MNIST / FMNIST
# ============================================================

if TASK == "MNIST":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,),
            (0.3081,),
        ),
    ])

    dataset_class = datasets.MNIST

elif TASK == "FMNIST":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.2860,),
            (0.3530,),
        ),
    ])

    dataset_class = datasets.FashionMNIST

else:
    raise ValueError(
        f"Unsupported task: {TASK}"
    )


train_dataset = dataset_class(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = dataset_class(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)


# Validate flattened input size
sample_image, _ = train_dataset[0]
dataset_input_size = int(sample_image.numel())

if args.archi[0] != dataset_input_size:
    raise ValueError(
        f"Architecture input size is {args.archi[0]}, "
        f"but {TASK} flattened input size is {dataset_input_size}"
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

class DirectCostFeedbackEquilibriumLearning:

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
                    args.act_clamp,
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
    # Direct supervised output teaching signal
    #
    # Full-output squared-error objective:
    #
    #   C = 1/2 ||y - t||^2
    #
    # where t is the one-hot target vector. Therefore
    #
    #   -dC/dy = t - y
    #
    # This full output teaching signal is propagated directly
    # through transpose feedback weights. There is no action
    # selection, no reward, no nudged phase, and no activation
    # derivative in the feedback propagation.
    # ========================================================

    @torch.no_grad()
    def output_cost_signal(
        self,
        output,
        labels,
    ):

        target = F.one_hot(
            labels,
            num_classes=self.architecture[-1],
        ).to(
            dtype=output.dtype,
            device=output.device,
        )

        # Negative gradient of C = 1/2 ||y - t||^2
        # with respect to the output activity y.
        return target - output


    # ========================================================
    # Direct cost-feedback update
    #
    # General arbitrary-depth version:
    #
    #   M_output = -dC/dy = target - output
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
        labels,
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
        # Output teaching signal: -dC/dy = target - output
        # ====================================================

        modulators[-1] = (
            self.output_cost_signal(
                output,
                labels,
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
    f"DirectCostFeedback_{TASK}"
    f"_arch{'-'.join(map(str, args.archi))}"
    f"_T1{T1}"
    f"_dt{DT}"
    f"_gamma{GAMMA}"
    f"_act{args.act}"
    f"_clamp{args.act_clamp if args.act_clamp is not None else 'none'}"
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
print("Direct Cost-Feedback Equilibrium Learning")
print("=" * 76)
print(f"Task           : {TASK}")
print(f"Architecture   : {args.archi}")
print(f"Hidden layers  : {NUM_HIDDEN_LAYERS}")
print(f"T1             : {T1}")
print(f"dt             : {DT}")
print(f"gamma          : {GAMMA}")
print(f"Hidden act     : {args.act}")
print(f"Hidden clamp   : {args.act_clamp}")
print(f"Output act     : {args.output_act}")
print("Teaching signal : -dC/dy = one_hot(target) - output")
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
    DirectCostFeedbackEquilibriumLearning(
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


        # 2. Prediction from the free-equilibrium output
        pred = torch.argmax(
            output,
            dim=1,
        )


        # 3. Direct full-output cost-feedback update
        #
        # C = 1/2 ||y - t||^2
        # -dC/dy = t - y
        model.update_weights(
            x,
            hidden_states,
            output,
            labels,
        )


        # Statistics
        run_correct += (
            pred == labels
        ).sum().item()


        run_total += (
            labels.size(0)
        )


    run_acc = (
        run_correct
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
        "Test Acc: {:.2f}%".format(
            epoch + 1,
            100.0 * run_acc,
            100.0 * test_acc_t,
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
