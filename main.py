import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithms.bcd_cyclic import CyclicBlockCoordinateDescent
from algorithms.bcd_gl import BlockCoordinateDescentGroupLasso
from algorithms.fista import FISTA

from dataset.yl1L import yl1LDataset
from dataset.yl4L import yl4LDataset
from dataset.ljyL import ljyLDataset
from dataset.mgb2L import mgb2LDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="yl1L",
    choices=["yl1L", "yl4L", "ljyL", "mgb2L"],
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="fista",
    choices=["fista", "bcd_gl", "bcd_cyclic"],
)
parser.add_argument("--lambda_", type=float, default=0.01)
parser.add_argument("--step_size", type=float, default=1e-4)
parser.add_argument("--random_seed", type=int, default=10)
parser.add_argument("--max_epochs", type=int, default=10)
parser.add_argument("--time_crit", type=float, default=250)
args = parser.parse_args()

np.random.seed(args.random_seed)

if args.dataset == "yl1L":
    dataset = yl1LDataset(5000)
elif args.dataset == "yl4L":
    dataset = yl4LDataset(2000)
elif args.dataset == "ljyL":
    dataset = ljyLDataset(2000)
elif args.dataset == "mgb2L":
    dataset = mgb2LDataset(2000)

design_matrix = dataset.design_matrix
response = dataset.response
x_star = dataset.x_star
block_indices = dataset.block_indices

print("Starting")

if args.algorithm == "bcd_cyclic":
    algorithm = CyclicBlockCoordinateDescent(
        block_indices,
        design_matrix,
        response,
        args.lambda_,
        step_size=args.step_size,
        time_crit=args.time_crit,
        # verbose=True,
    )
    final_iterator, cache = algorithm.blockCoordinateDescent(args.max_epochs)
elif args.algorithm == "bcd_gl":
    algorithm = BlockCoordinateDescentGroupLasso(
        block_indices,
        design_matrix,
        response,
        args.lambda_,
        time_crit=args.time_crit,
    )
    final_iterator, cache = algorithm.blockCoordinateDescent(args.max_epochs)
elif args.algorithm == "fista":
    algorithm = FISTA(
        block_indices,
        design_matrix,
        response,
        args.lambda_,
        time_crit=args.time_crit,
    )
    final_iterator, cache = algorithm.fista(args.max_epochs)


losses = cache["losses"]
time_itr = cache["time_itr"]
final_loss = cache["final_loss"]

desc = {
    "losses": losses,
    "time_itr": time_itr,
    "lambda_": args.lambda_,
    "step_size": args.step_size,
    "preprocess_time": algorithm.preprocess_time,
}

os.makedirs(f"./results/{args.algorithm}/{args.dataset}", exist_ok=True)
pd.DataFrame(desc).to_csv(
    f"./results/{args.algorithm}/{args.dataset}/results.csv", index=False
)

time_itr = np.cumsum(time_itr)
plt.plot(time_itr, losses)
plt.xlabel("Time (s)")
plt.ylabel("Loss")
plt.title(f"{args.algorithm} ({args.dataset}) ({final_loss:.2f})")
plt.savefig(f"./results/{args.algorithm}/{args.dataset}/results.png")

plt.clf()
plt.plot(time_itr, losses)
plt.xlabel("Time (s)")
plt.ylabel("Loss")
plt.yscale("log")
plt.title(f"{args.algorithm} ({args.dataset}) ({final_loss:.2f})")
plt.savefig(f"./results/{args.algorithm}/{args.dataset}/results_logscale.png")

print("Total Time Taken:", time_itr[-1] + algorithm.preprocess_time)

args_dict = vars(args)
with open(f"./results/{args.algorithm}/{args.dataset}/args.json", "w") as f:
    json.dump(args_dict, f)
