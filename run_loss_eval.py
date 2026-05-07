"""Standalone loss eval runner — used to redo step 5 at a larger batch size
than the pipeline default of 2 (which is wasteful on a 31B / H200).

Loads the direction selected by the main pipeline and runs evaluate_loss for
baseline / ablation / actadd interventions, writing into the same
runs/{alias}/loss_evals/ directory the pipeline would have used.
"""
import argparse
import json
import os

import torch

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import (
    get_activation_addition_input_pre_hook,
    get_all_direction_ablation_hooks,
)
from pipeline.submodules.evaluate_loss import evaluate_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument(
        "--n_batches",
        type=int,
        default=None,
        help="Override n_batches; defaults to cfg.ce_loss_n_batches // (batch_size/2) so we evaluate the same total token count as the pipeline default.",
    )
    args = ap.parse_args()

    model_alias = os.path.basename(args.model_path)
    cfg = Config(model_alias=model_alias, model_path=args.model_path)

    # Default n_batches scales inversely with batch_size to preserve total tokens.
    n_batches = args.n_batches
    if n_batches is None:
        baseline_total = cfg.ce_loss_batch_size * cfg.ce_loss_n_batches
        n_batches = max(1, baseline_total // args.batch_size)

    artifact = cfg.artifact_path()
    direction = torch.load(f"{artifact}/direction.pt", weights_only=False)
    with open(f"{artifact}/direction_metadata.json") as f:
        meta = json.load(f)
    layer = meta["layer"]
    print(f"[loss-eval] direction.pt loaded; pos={meta['pos']} layer={layer}")
    print(f"[loss-eval] batch_size={args.batch_size} n_batches={n_batches} "
          f"(pipeline default was {cfg.ce_loss_batch_size}x{cfg.ce_loss_n_batches})")

    model_base = construct_model_base(args.model_path)
    direction = direction.to(model_base.model.device)

    out_dir = os.path.join(artifact, "loss_evals")
    os.makedirs(out_dir, exist_ok=True)

    completions_path = os.path.join(artifact, "completions", "harmless_baseline_completions.json")
    if not os.path.exists(completions_path):
        raise SystemExit(
            f"[loss-eval] missing {completions_path}; the pipeline must finish step 4a before this script can run."
        )

    interventions = {
        "baseline": ([], []),
        "ablation": get_all_direction_ablation_hooks(model_base, direction),
        "actadd": (
            [
                (
                    model_base.model_block_modules[layer],
                    get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0),
                )
            ],
            [],
        ),
    }

    for label, (pre, post) in interventions.items():
        print(f"\n[loss-eval] === {label} ===")
        result = evaluate_loss(
            model_base,
            pre,
            post,
            batch_size=args.batch_size,
            n_batches=n_batches,
            completions_file_path=completions_path,
        )
        out_path = os.path.join(out_dir, f"{label}_loss_eval.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=4)
        print(f"[loss-eval] wrote {out_path}")


if __name__ == "__main__":
    main()
