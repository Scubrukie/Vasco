from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .data import (
        CONDITION_MASKS,
        CONDITION_ORDER,
        MISSING_ORDER,
        MISSING_TO_CONDITION,
        MISSING_TO_INDEX,
        PackedData,
        collate_sequences,
        make_projector_collate,
        make_selector_collate,
        observed_mask,
        torch_load,
    )
    from .model import (
        ActionSelector,
        Predictor,
        PredictorConfig,
        ProjectorConfig,
        SelectorConfig,
        SemanticProjector,
    )
except ImportError:
    from data import (
        CONDITION_MASKS,
        CONDITION_ORDER,
        MISSING_ORDER,
        MISSING_TO_CONDITION,
        MISSING_TO_INDEX,
        PackedData,
        collate_sequences,
        make_projector_collate,
        make_selector_collate,
        observed_mask,
        torch_load,
    )
    from model import (
        ActionSelector,
        Predictor,
        PredictorConfig,
        ProjectorConfig,
        SelectorConfig,
        SemanticProjector,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def parse_names(value: str, allowed: Iterable[str]) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(names) - set(allowed))
    if not names or unknown:
        raise ValueError(f"invalid names {value!r}; unknown={unknown}")
    return names


def parse_weights(value: str, names: Iterable[str], default: float = 1.0) -> dict[str, float]:
    result = {name: float(default) for name in names}
    if not value.strip():
        return result
    for item in value.split(","):
        name, separator, raw_weight = item.strip().partition("=")
        if not separator or name not in result:
            raise ValueError(f"invalid weight specification: {item!r}")
        result[name] = float(raw_weight)
    return result


def condition_tensor(name: str, device: torch.device) -> torch.Tensor:
    return torch.tensor(CONDITION_MASKS[name], device=device, dtype=torch.long)


def weighted_f1(prediction: np.ndarray, label: np.ndarray) -> float:
    predicted = prediction > 0
    target = label > 0
    total = max(target.size, 1)
    result = 0.0
    for positive in (False, True):
        true_positive = np.logical_and(predicted == positive, target == positive).sum()
        false_positive = np.logical_and(predicted == positive, target != positive).sum()
        false_negative = np.logical_and(predicted != positive, target == positive).sum()
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0 else 2 * true_positive / denominator
        result += (target == positive).sum() / total * f1
    return float(result)


def compute_metrics(
    prediction: list[torch.Tensor],
    label: list[torch.Tensor],
) -> dict[str, float]:
    if not prediction:
        return {"n": 0, "acc2": 0.0, "weighted_f1": 0.0, "mae": 0.0, "corr": 0.0}
    pred = torch.cat(prediction).float().cpu().numpy()
    gold = torch.cat(label).float().cpu().numpy()
    nonzero = gold != 0
    pred = pred[nonzero]
    gold = gold[nonzero]
    if gold.size == 0:
        return {"n": 0, "acc2": 0.0, "weighted_f1": 0.0, "mae": 0.0, "corr": 0.0}
    accuracy = float(((pred > 0) == (gold > 0)).mean())
    mae = float(np.abs(pred - gold).mean())
    corr = 0.0
    if pred.size > 1 and np.std(pred) > 0 and np.std(gold) > 0:
        corr = float(np.corrcoef(pred, gold)[0, 1])
    return {
        "n": int(gold.size),
        "acc2": accuracy,
        "weighted_f1": weighted_f1(pred, gold),
        "mae": mae,
        "corr": corr,
    }


def mean_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = ("acc2", "weighted_f1", "mae", "corr")
    return {
        key: float(np.mean([values[key] for values in metrics.values()]))
        for key in keys
    }


def metric_score(metrics: dict[str, float]) -> float:
    return float(metrics["acc2"] + metrics["weighted_f1"])


def print_metrics(title: str, metrics: dict[str, dict[str, float]]) -> None:
    print(f"\n== {title} ==")
    for name, values in metrics.items():
        print(
            f"{name:12s} Acc-2={100 * values['acc2']:6.2f} "
            f"WF1={100 * values['weighted_f1']:6.2f}"
        )


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def checkpoint_args(args: argparse.Namespace) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in vars(args).items():
        if callable(value):
            continue
        output[key] = str(value) if isinstance(value, Path) else value
    return output


def compatible_state(module: torch.nn.Module, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    expected = module.state_dict()
    return {
        key: value
        for key, value in state.items()
        if key in expected and expected[key].shape == value.shape
    }


def load_predictor(path: str | Path, device: torch.device) -> Predictor:
    checkpoint = torch_load(path, trusted_checkpoint=True)
    raw_config = checkpoint.get("predictor_config", checkpoint.get("config", {}))
    config = PredictorConfig(
        d_model=int(raw_config.get("d_model", 1024)),
        dropout=float(raw_config.get("dropout", 0.3)),
        context_layers=int(raw_config.get("context_layers", 1)),
        use_condition_emb=bool(raw_config.get("use_condition_emb", True)),
    )
    model = Predictor(config).to(device)
    state = checkpoint.get("predictor", checkpoint.get("model", checkpoint.get("state_dict")))
    if state is None:
        raise KeyError(f"{path}: checkpoint has no predictor/model/state_dict")
    usable = compatible_state(model, state)
    result = model.load_state_dict(usable, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(
            f"loaded predictor non-strictly: missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def predictor_loss(
    prediction: torch.Tensor,
    label: torch.Tensor,
    mask: torch.Tensor,
    *,
    lambda_cls: float,
    lambda_sign: float,
    sign_margin: float,
) -> torch.Tensor:
    active = mask & label.ne(0)
    weight = active.to(prediction.dtype)
    mse = ((prediction - label).pow(2) * weight).sum() / weight.sum().clamp_min(1.0)
    if not bool(active.any().item()):
        return mse
    target = label.gt(0).to(prediction.dtype)
    classification = F.binary_cross_entropy_with_logits(
        prediction[active],
        target[active],
    )
    sign = torch.where(label.gt(0), 1.0, -1.0).to(prediction.dtype)
    sign_loss = F.relu(sign_margin - sign[active] * prediction[active]).mean()
    return mse + lambda_cls * classification + lambda_sign * sign_loss


def evaluate_predictor(
    model: Predictor,
    loader: DataLoader,
    device: torch.device,
    conditions: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    model.eval()
    store: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"prediction": [], "label": []} for name in conditions
    }
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            active = batch["mask"] & batch["labels"].ne(0)
            for name in conditions:
                output = model(
                    batch["audio"],
                    batch["vision"],
                    batch["text"],
                    condition_tensor(name, device),
                    batch["mask"],
                )["pred"]
                store[name]["prediction"].append(output[active].cpu())
                store[name]["label"].append(batch["labels"][active].cpu())
    metrics = {
        name: compute_metrics(values["prediction"], values["label"])
        for name, values in store.items()
    }
    metrics["Mean"] = {**mean_metrics(metrics), "n": sum(v["n"] for v in metrics.values())}
    return metrics


def train_predictor(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    package = PackedData(args.data)
    conditions = parse_names(args.conditions, CONDITION_ORDER)
    condition_weights = parse_weights(args.condition_weights, conditions)
    train_loader = DataLoader(
        package.sequence_dataset("train"),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_sequences,
    )
    test_loader = DataLoader(
        package.sequence_dataset("test"),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sequences,
    )
    config = PredictorConfig(
        d_model=args.d_model,
        dropout=args.dropout,
        context_layers=args.context_layers,
    )
    model = Predictor(config).to(device)
    if args.init_checkpoint:
        source = torch_load(args.init_checkpoint, trusted_checkpoint=True)
        state = source.get("predictor", source.get("model", source.get("state_dict")))
        if state is None:
            raise KeyError(f"{args.init_checkpoint}: missing model state")
        result = model.load_state_dict(compatible_state(model, state), strict=False)
        print(
            f"initialized predictor: missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_epochs = {int(value) for value in args.save_epochs.split(",") if value.strip()}
    history = []
    best_score = -math.inf
    best_row: dict[str, Any] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            condition_losses = []
            weights = []
            for name in conditions:
                prediction = model(
                    batch["audio"],
                    batch["vision"],
                    batch["text"],
                    condition_tensor(name, device),
                    batch["mask"],
                )["pred"]
                condition_losses.append(
                    predictor_loss(
                        prediction,
                        batch["labels"],
                        batch["mask"],
                        lambda_cls=args.lambda_cls,
                        lambda_sign=args.lambda_sign,
                        sign_margin=args.sign_margin,
                    )
                )
                weights.append(condition_weights[name])
            loss_weights = torch.tensor(weights, device=device)
            loss = (
                torch.stack(condition_losses) * loss_weights
            ).sum() / loss_weights.sum().clamp_min(1e-12)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate_predictor(model, test_loader, device, conditions)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "metrics": metrics,
        }
        history.append(row)
        score = metrics["Mean"]["acc2"]
        checkpoint = {
            "stage": "predictor",
            "epoch": epoch,
            "predictor": model.state_dict(),
            "predictor_config": model.checkpoint_config(),
            "args": checkpoint_args(args),
            "metrics": metrics,
            "best": best_row,
        }
        if score > best_score:
            best_score = score
            best_row = row
            checkpoint["best"] = best_row
            torch.save(checkpoint, out_dir / "best.pt")
        if epoch in save_epochs:
            torch.save(checkpoint, out_dir / f"epoch_{epoch:03d}.pt")
        save_json(out_dir / "history.json", history)
        print_metrics(f"predictor epoch {epoch}", metrics)
    if best_row is not None:
        save_json(out_dir / "best_metrics.json", best_row)


def projector_forward(
    predictor: Predictor,
    projector: SemanticProjector,
    batch: dict[str, Any],
    task_weights: dict[str, float],
    lambda_task: float,
    lambda_direction: float,
) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    batch_size, length = batch["mask"].shape
    condition = observed_mask(batch["missing_idx"], length).to(batch["audio"].device)
    full_condition = torch.ones_like(condition)
    active = batch["mask"] & batch["action_mask"] & batch["labels"].ne(0)
    with torch.no_grad():
        incomplete = predictor(
            batch["audio"],
            batch["vision"],
            batch["text"],
            condition,
            batch["mask"],
        )
        complete = predictor(
            batch["audio"],
            batch["vision"],
            batch["text"],
            full_condition,
            batch["mask"],
        )
    flat_active = active.reshape(-1)
    if not bool(flat_active.any().item()):
        zero = next(projector.parameters()).sum() * 0.0
        return zero, {}
    missing = batch["missing_idx"][:, None].expand(batch_size, length).reshape(-1)
    state = incomplete["state"].reshape(batch_size * length, -1)
    action = batch["action"].reshape(batch_size * length, -1)
    anchors = batch["anchors"].reshape(-1)
    output = projector(
        action[flat_active],
        missing[flat_active],
        anchors[flat_active],
        state[flat_active],
    )
    updated = state[flat_active] + output["delta"]
    prediction = predictor.predict(updated)
    label = batch["labels"].reshape(-1)[flat_active]
    row_weights = prediction.new_tensor(
        [task_weights[MISSING_ORDER[int(index)]] for index in missing[flat_active]]
    )
    task = (
        (prediction - label).pow(2) * row_weights
    ).sum() / row_weights.sum().clamp_min(1.0)
    target = complete["state"].reshape(batch_size * length, -1)[flat_active] - state[flat_active]
    direction = (
        1.0
        - F.cosine_similarity(
            output["direction"],
            F.normalize(target, dim=-1, eps=1e-6),
            dim=-1,
            eps=1e-6,
        )
    ).mean()
    records: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    for missing_name, missing_value in MISSING_TO_INDEX.items():
        selected = missing[flat_active].eq(missing_value)
        if bool(selected.any().item()):
            records[missing_name] = (
                [prediction[selected].detach().cpu()],
                [label[selected].detach().cpu()],
            )
    return lambda_task * task + lambda_direction * direction, records


def run_projector_epoch(
    predictor: Predictor,
    projector: SemanticProjector,
    loader: DataLoader,
    device: torch.device,
    task_weights: dict[str, float],
    lambda_task: float,
    lambda_direction: float,
    optimizer: torch.optim.Optimizer | None,
    max_grad_norm: float,
) -> tuple[float, dict[str, dict[str, float]]]:
    projector.train(optimizer is not None)
    losses = []
    store: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: {"prediction": [], "label": []}
    )
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, records = projector_forward(
                predictor,
                projector,
                batch,
                task_weights,
                lambda_task,
                lambda_direction,
            )
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(projector.parameters(), max_grad_norm)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for name, (prediction, label) in records.items():
                store[name]["prediction"].extend(prediction)
                store[name]["label"].extend(label)
    metrics = {
        name: compute_metrics(values["prediction"], values["label"])
        for name, values in store.items()
    }
    return float(np.mean(losses)), metrics


def save_projector_checkpoint(
    path: Path,
    projector: SemanticProjector,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, dict[str, float]],
) -> None:
    torch.save(
        {
            "stage": "projector",
            "epoch": epoch,
            "projector": projector.state_dict(),
            "projector_config": projector.checkpoint_config(),
            "args": checkpoint_args(args),
            "metrics": metrics,
        },
        path,
    )


def train_projector(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    package = PackedData(args.data)
    missing = parse_names(args.missing, MISSING_ORDER)
    task_weights = parse_weights(args.task_weights, MISSING_ORDER)
    predictor = load_predictor(args.predictor_checkpoint, device)
    train_loader = DataLoader(
        package.projector_dataset("train", missing),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_projector_collate(package),
    )
    test_loader = DataLoader(
        package.projector_dataset("test", missing),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_projector_collate(package),
    )
    candidate_counts = {
        int(package.action_split("train")[name]["action_ids"].size(1))
        for name in missing
    }
    if len(candidate_counts) != 1:
        raise ValueError("projector run cannot mix conditions with different candidate counts")
    num_candidates = candidate_counts.pop()
    config = ProjectorConfig(
        text_dim=package.action_dim,
        d_model=predictor.config.d_model,
        num_anchors=num_candidates,
        num_missing=max(3, max(MISSING_TO_INDEX[name] for name in missing) + 1),
        bottleneck=args.bottleneck,
        meta_dim=args.meta_dim,
        dropout=args.dropout,
        init_alpha=args.init_alpha,
        max_alpha=args.max_alpha,
    )
    projector = SemanticProjector(config).to(device)
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {name: -math.inf for name in missing}
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_projector_epoch(
            predictor,
            projector,
            train_loader,
            device,
            task_weights,
            args.lambda_task,
            args.lambda_direction,
            optimizer,
            args.max_grad_norm,
        )
        test_loss, metrics = run_projector_epoch(
            predictor,
            projector,
            test_loader,
            device,
            task_weights,
            args.lambda_task,
            args.lambda_direction,
            None,
            args.max_grad_norm,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "metrics": metrics,
        }
        history.append(row)
        for name in missing:
            score = metric_score(metrics[name])
            if score > best[name]:
                best[name] = score
                save_projector_checkpoint(
                    out_dir / f"best_{name}.pt",
                    projector,
                    args,
                    epoch,
                    metrics,
                )
        target = args.select_target or missing[0]
        if metric_score(metrics[target]) >= best[target]:
            save_projector_checkpoint(
                out_dir / "best.pt",
                projector,
                args,
                epoch,
                metrics,
            )
        save_json(out_dir / "history.json", history)
        print(f"projector epoch {epoch}", flush=True)


def load_projector(path: str | Path, device: torch.device) -> SemanticProjector:
    checkpoint = torch_load(path, trusted_checkpoint=True)
    config = ProjectorConfig(**checkpoint["projector_config"])
    model = SemanticProjector(config).to(device)
    model.load_state_dict(checkpoint["projector"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def mask_q(
    values: torch.Tensor,
    used: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    output = values.clone()
    output[:, :-1] = output[:, :-1].masked_fill(
        used | ~candidate_mask,
        torch.finfo(output.dtype).min,
    )
    return output


def semantic_gain(
    current_delta: torch.Tensor,
    next_delta: torch.Tensor,
    target_delta: torch.Tensor,
) -> torch.Tensor:
    current = F.cosine_similarity(current_delta, target_delta, dim=-1, eps=1e-8)
    following = F.cosine_similarity(
        next_delta,
        target_delta[:, None, :],
        dim=-1,
        eps=1e-8,
    )
    return following - current[:, None]


def next_max_q(
    selector: ActionSelector,
    projector: SemanticProjector,
    actions: torch.Tensor,
    anchors: torch.Tensor,
    candidate_mask: torch.Tensor,
    missing: torch.Tensor,
    state: torch.Tensor,
    deltas: torch.Tensor,
    used: torch.Tensor,
    step: int,
    disable_stop: bool,
) -> torch.Tensor:
    batch, candidates, action_dim = actions.shape
    next_state = state[:, None, :] + deltas
    flat_state = next_state.reshape(batch * candidates, -1)
    flat_actions = actions[:, None, :, :].expand(
        -1, candidates, -1, -1
    ).reshape(batch * candidates, candidates, action_dim)
    flat_anchors = anchors[:, None, :].expand(-1, candidates, -1).reshape(
        batch * candidates, candidates
    )
    flat_candidate = candidate_mask[:, None, :].expand(
        -1, candidates, -1
    ).reshape(batch * candidates, candidates)
    flat_missing = missing[:, None].expand(-1, candidates).reshape(-1)
    flat_used = used[:, None, :].expand(-1, candidates, -1).clone()
    rows = torch.arange(batch, device=used.device)[:, None].expand(-1, candidates)
    chosen = torch.arange(candidates, device=used.device)[None, :].expand(batch, -1)
    flat_used[rows, chosen, chosen] = True
    flat_used = flat_used.reshape(batch * candidates, candidates)
    _, future_latent = projector.all_candidates(
        flat_actions,
        flat_missing,
        flat_anchors,
        flat_state,
    )
    future_q = selector(
        state=flat_state,
        action_latent=future_latent,
        missing_idx=flat_missing,
        used_mask=flat_used,
        candidate_mask=flat_candidate,
        step=step,
    )
    future_q = mask_q(future_q, flat_used, flat_candidate)
    if disable_stop:
        has_action = (flat_candidate & ~flat_used).any(dim=1)
        future_q[has_action, -1] = torch.finfo(future_q.dtype).min
    return future_q.max(dim=1).values.view(batch, candidates)


def selector_batch(
    predictor: Predictor,
    projector: SemanticProjector,
    selector: ActionSelector,
    batch: dict[str, Any],
    args: argparse.Namespace,
    policy: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    batch_size, length = batch["mask"].shape
    condition = observed_mask(batch["missing_idx"], length).to(batch["audio"].device)
    full_condition = torch.ones_like(condition)
    active = batch["mask"] & batch["action_mask"] & batch["labels"].ne(0)
    with torch.no_grad():
        incomplete = predictor(
            batch["audio"],
            batch["vision"],
            batch["text"],
            condition,
            batch["mask"],
        )
        complete = predictor(
            batch["audio"],
            batch["vision"],
            batch["text"],
            full_condition,
            batch["mask"],
        )
    flat_active = active.reshape(-1)
    if not bool(flat_active.any().item()):
        zero = next(selector.parameters()).sum() * 0.0
        return zero, {}
    candidates = batch["actions"].size(2)
    actions = batch["actions"].reshape(batch_size * length, candidates, -1)[flat_active]
    anchors = batch["anchors"].reshape(batch_size * length, candidates)[flat_active]
    candidate_mask = batch["candidate_mask"].reshape(
        batch_size * length, candidates
    )[flat_active]
    missing = batch["missing_idx"][:, None].expand(batch_size, length).reshape(-1)[
        flat_active
    ]
    labels = batch["labels"].reshape(-1)[flat_active]
    initial_state = incomplete["state"].reshape(batch_size * length, -1)[flat_active]
    full_state = complete["state"].reshape(batch_size * length, -1)[flat_active]
    initial_prediction = incomplete["pred"].reshape(-1)[flat_active]
    target_delta = full_state - initial_state
    state = initial_state.clone()
    used = torch.zeros_like(candidate_mask)
    done = torch.zeros(labels.size(0), device=labels.device, dtype=torch.bool)
    chain_length = torch.zeros(labels.size(0), device=labels.device)
    q_losses: list[torch.Tensor] = []
    task_weights = labels.new_tensor(
        [args.task_weight_map[MISSING_ORDER[int(value)]] for value in missing]
    )
    semantic_weights = labels.new_tensor(
        [args.semantic_weight_map[MISSING_ORDER[int(value)]] for value in missing]
    )
    rollout_steps = min(args.rollout_steps, args.max_steps)
    for step in range(rollout_steps):
        alive = ~done
        if not bool(alive.any().item()):
            break
        with torch.no_grad():
            deltas, latents = projector.all_candidates(
                actions,
                missing,
                anchors,
                state,
            )
            prediction_before = predictor.predict(state)
            next_state = state[:, None, :] + deltas
            prediction_after = predictor.predict(next_state)
            current_loss = (prediction_before - labels).pow(2)
            next_loss = (prediction_after - labels[:, None]).pow(2)
            task_gain = current_loss[:, None] - next_loss
            current_delta = state - initial_state
            following_delta = next_state - initial_state[:, None, :]
            direction_gain = semantic_gain(
                current_delta,
                following_delta,
                target_delta,
            )
            action_reward = (
                task_weights[:, None] * task_gain
                + args.lambda_semantic_reward
                * semantic_weights[:, None]
                * direction_gain
                - args.step_cost
            )
            rewards = torch.cat(
                [action_reward, torch.zeros_like(labels)[:, None]],
                dim=1,
            )
        q_values = selector(
            state=state,
            action_latent=latents.detach(),
            missing_idx=missing,
            used_mask=used,
            candidate_mask=candidate_mask,
            step=step,
        )
        masked_q = mask_q(q_values, used, candidate_mask)
        if args.disable_stop:
            has_action = (candidate_mask & ~used).any(dim=1) & alive
            masked_q[has_action, -1] = torch.finfo(masked_q.dtype).min
        masked_q = masked_q.masked_fill(done[:, None], 0.0)
        with torch.no_grad():
            target = rewards.clone()
            if step + 1 < rollout_steps and args.gamma:
                target[:, :-1] = action_reward + args.gamma * next_max_q(
                    selector,
                    projector,
                    actions,
                    anchors,
                    candidate_mask,
                    missing,
                    state,
                    deltas,
                    used,
                    step + 1,
                    args.disable_stop,
                )
            valid_action = candidate_mask & ~used
            valid = torch.cat(
                [valid_action, torch.ones_like(done)[:, None]],
                dim=1,
            ) & alive[:, None]
        if bool(valid.any().item()):
            weights = task_weights[:, None].expand_as(q_values)
            error = F.smooth_l1_loss(q_values, target, reduction="none")
            denominator = weights[valid].sum()
            if float(denominator) > 0:
                q_losses.append((error[valid] * weights[valid]).sum() / denominator)

        if policy == "random_one" or policy == "random_k":
            valid_action = (candidate_mask & ~used) & alive[:, None]
            random_score = torch.rand_like(q_values[:, :-1]).masked_fill(
                ~valid_action, -1.0
            )
            chosen = random_score.argmax(dim=1)
            has_action = valid_action.any(dim=1)
            chosen = torch.where(
                has_action,
                chosen,
                torch.full_like(chosen, candidates),
            )
        else:
            chosen = masked_q.argmax(dim=1)
        if policy == "random_one" and step >= 0:
            force_end_after_step = True
        else:
            force_end_after_step = False
        chosen = torch.where(
            alive,
            chosen,
            torch.full_like(chosen, candidates),
        )
        stop = chosen.eq(candidates)
        action_index = chosen.clamp(max=candidates - 1)
        selected_delta = deltas[
            torch.arange(deltas.size(0), device=deltas.device),
            action_index,
        ]
        apply = alive & ~stop
        state = state + selected_delta * apply[:, None].to(state.dtype)
        used[
            torch.arange(used.size(0), device=used.device),
            action_index,
        ] |= apply
        chain_length += apply.float()
        done |= alive & stop
        if force_end_after_step:
            done |= alive
    final_prediction = predictor.predict(state)
    q_loss = (
        torch.stack(q_losses).mean()
        if q_losses
        else next(selector.parameters()).sum() * 0.0
    )
    records = {
        "prediction": final_prediction.detach().cpu(),
        "initial_prediction": initial_prediction.detach().cpu(),
        "label": labels.detach().cpu(),
        "missing": missing.detach().cpu(),
        "chain_length": chain_length.detach().cpu(),
    }
    return q_loss, records


def run_selector_epoch(
    predictor: Predictor,
    projector: SemanticProjector,
    selector: ActionSelector,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
    policy: str,
) -> tuple[float, dict[str, dict[str, float]], float]:
    selector.train(optimizer is not None)
    losses = []
    store: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: {"prediction": [], "label": []}
    )
    chain_lengths: list[torch.Tensor] = []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, records = selector_batch(
                predictor,
                projector,
                selector,
                batch,
                args,
                policy,
            )
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(selector.parameters(), args.max_grad_norm)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if records:
                chain_lengths.append(records["chain_length"])
                for missing_value, missing_name in enumerate(MISSING_ORDER):
                    selected = records["missing"].eq(missing_value)
                    if bool(selected.any().item()):
                        store[missing_name]["prediction"].append(
                            records["prediction"][selected]
                        )
                        store[missing_name]["label"].append(records["label"][selected])
    metrics = {
        name: compute_metrics(values["prediction"], values["label"])
        for name, values in store.items()
    }
    chain = (
        float(torch.cat(chain_lengths).mean())
        if chain_lengths
        else 0.0
    )
    return float(np.mean(losses)), metrics, chain


def save_selector_checkpoint(
    path: Path,
    predictor: Predictor,
    projector: SemanticProjector,
    selector: ActionSelector,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, dict[str, float]],
    chain_length: float,
) -> None:
    torch.save(
        {
            "stage": "selector",
            "epoch": epoch,
            "predictor": predictor.state_dict(),
            "predictor_config": predictor.checkpoint_config(),
            "projector": projector.state_dict(),
            "projector_config": projector.checkpoint_config(),
            "selector": selector.state_dict(),
            "selector_config": selector.checkpoint_config(),
            "args": {
                **checkpoint_args(args),
                "task_weight_map": dict(args.task_weight_map),
                "semantic_weight_map": dict(args.semantic_weight_map),
            },
            "metrics": metrics,
            "chain_length": chain_length,
        },
        path,
    )


def train_selector(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    package = PackedData(args.data)
    missing = parse_names(args.missing, MISSING_ORDER)
    args.task_weight_map = parse_weights(args.task_weights, MISSING_ORDER)
    args.semantic_weight_map = parse_weights(args.semantic_weights, MISSING_ORDER)
    predictor = load_predictor(args.predictor_checkpoint, device)
    projector = load_projector(args.projector_checkpoint, device)
    train_loader = DataLoader(
        package.selector_dataset("train", missing),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_selector_collate(package),
    )
    test_loader = DataLoader(
        package.selector_dataset("test", missing),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_selector_collate(package),
    )
    candidate_counts = {
        int(package.action_split("train")[name]["action_ids"].size(1))
        for name in missing
    }
    if len(candidate_counts) != 1:
        raise ValueError("selector run cannot mix conditions with different candidate counts")
    config = SelectorConfig(
        d_model=predictor.config.d_model,
        action_dim=projector.config.bottleneck,
        num_candidates=candidate_counts.pop(),
        num_missing=projector.config.num_missing,
        max_steps=args.max_steps,
        meta_dim=args.meta_dim,
        hidden=args.hidden,
        dropout=args.dropout,
    )
    selector = ActionSelector(config).to(device)
    optimizer = torch.optim.AdamW(
        selector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = args.select_target or missing[0]
    best_score = -math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_selector_epoch(
            predictor,
            projector,
            selector,
            train_loader,
            device,
            args,
            optimizer,
            "q",
        )
        test_loss, metrics, chain = run_selector_epoch(
            predictor,
            projector,
            selector,
            test_loader,
            device,
            args,
            None,
            "q",
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "chain_length": chain,
            "metrics": metrics,
        }
        history.append(row)
        score = metric_score(metrics[target])
        if score > best_score:
            best_score = score
            save_selector_checkpoint(
                out_dir / "best.pt",
                predictor,
                projector,
                selector,
                args,
                epoch,
                metrics,
                chain,
            )
        save_json(out_dir / "history.json", history)
        print_metrics(f"selector epoch {epoch}", metrics)


def load_selector_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[Predictor, SemanticProjector, ActionSelector, dict[str, Any]]:
    checkpoint = torch_load(path, trusted_checkpoint=True)
    predictor = Predictor(PredictorConfig(**checkpoint["predictor_config"])).to(device)
    predictor.load_state_dict(checkpoint["predictor"], strict=True)
    projector = SemanticProjector(
        ProjectorConfig(**checkpoint["projector_config"])
    ).to(device)
    projector.load_state_dict(checkpoint["projector"], strict=True)
    selector = ActionSelector(SelectorConfig(**checkpoint["selector_config"])).to(device)
    selector.load_state_dict(checkpoint["selector"], strict=True)
    for module in (predictor, projector, selector):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return predictor, projector, selector, checkpoint


def evaluate_selector(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    package = PackedData(args.data)
    predictor, projector, selector, checkpoint = load_selector_checkpoint(
        args.checkpoint,
        device,
    )
    stored_args = checkpoint["args"]
    missing = parse_names(args.missing, MISSING_ORDER)
    args.task_weight_map = stored_args["task_weight_map"]
    args.semantic_weight_map = stored_args["semantic_weight_map"]
    args.gamma = float(stored_args["gamma"])
    args.lambda_semantic_reward = float(stored_args["lambda_semantic_reward"])
    args.step_cost = float(stored_args["step_cost"])
    args.max_steps = int(stored_args["max_steps"])
    if args.rollout_steps is None:
        args.rollout_steps = args.max_steps
    loader = DataLoader(
        package.selector_dataset(args.split, missing),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_selector_collate(package),
    )
    _, metrics, chain = run_selector_epoch(
        predictor,
        projector,
        selector,
        loader,
        device,
        args,
        None,
        args.policy,
    )
    result = {
        "variant": args.variant,
        "split": args.split,
        "policy": args.policy,
        "rollout_steps": args.rollout_steps,
        "disable_stop": args.disable_stop,
        "chain_length": chain,
        "metrics": metrics,
    }
    print_metrics(args.variant, metrics)
    if args.metrics_json:
        save_json(Path(args.metrics_json), result)


def inspect_data(args: argparse.Namespace) -> None:
    package = PackedData(args.data)
    print(json.dumps(package.meta, indent=2))
    print(f"action vocabulary: {tuple(package.action_embeddings.shape)}")
    for split, values in package.features.items():
        print(
            f"features/{split}: samples={len(values['ids'])} "
            f"sequences={len(values['video_ids'])}"
        )
    for split, conditions in package.actions.items():
        for missing, group in conditions.items():
            print(
                f"actions/{split}/{missing}: groups={len(group['ids'])} "
                f"candidates={group['action_ids'].size(1)}"
            )


def add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compact TSA-HC training entrypoint")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    predictor = subparsers.add_parser("predictor")
    add_shared(predictor)
    predictor.add_argument("--out-dir", required=True)
    predictor.add_argument("--init-checkpoint")
    predictor.add_argument("--conditions", default=",".join(CONDITION_ORDER))
    predictor.add_argument("--condition-weights", default="")
    predictor.add_argument("--d-model", type=int, default=1024)
    predictor.add_argument("--context-layers", type=int, default=1)
    predictor.add_argument("--dropout", type=float, default=0.3)
    predictor.add_argument("--lambda-cls", type=float, default=0.3)
    predictor.add_argument("--lambda-sign", type=float, default=0.2)
    predictor.add_argument("--sign-margin", type=float, default=0.2)
    predictor.add_argument("--epochs", type=int, default=80)
    predictor.add_argument("--batch-size", type=int, default=64)
    predictor.add_argument("--eval-batch-size", type=int, default=64)
    predictor.add_argument("--lr", type=float, default=5e-5)
    predictor.add_argument("--weight-decay", type=float, default=1e-4)
    predictor.add_argument("--max-grad-norm", type=float, default=5.0)
    predictor.add_argument("--save-epochs", default="")
    predictor.set_defaults(function=train_predictor)

    projector = subparsers.add_parser("projector")
    add_shared(projector)
    projector.add_argument("--predictor-checkpoint", required=True)
    projector.add_argument("--out-dir", required=True)
    projector.add_argument("--missing", required=True)
    projector.add_argument("--task-weights", default="")
    projector.add_argument("--select-target")
    projector.add_argument("--bottleneck", type=int, default=256)
    projector.add_argument("--meta-dim", type=int, default=128)
    projector.add_argument("--dropout", type=float, default=0.1)
    projector.add_argument("--init-alpha", type=float, default=0.1)
    projector.add_argument("--max-alpha", type=float, default=0.7)
    projector.add_argument("--lambda-task", type=float, default=1.0)
    projector.add_argument("--lambda-direction", type=float, default=0.2)
    projector.add_argument("--epochs", type=int, default=60)
    projector.add_argument("--batch-size", type=int, default=16)
    projector.add_argument("--eval-batch-size", type=int, default=32)
    projector.add_argument("--lr", type=float, default=1e-4)
    projector.add_argument("--weight-decay", type=float, default=1e-4)
    projector.add_argument("--max-grad-norm", type=float, default=5.0)
    projector.set_defaults(function=train_projector)

    selector = subparsers.add_parser("selector")
    add_shared(selector)
    selector.add_argument("--predictor-checkpoint", required=True)
    selector.add_argument("--projector-checkpoint", required=True)
    selector.add_argument("--out-dir", required=True)
    selector.add_argument("--missing", required=True)
    selector.add_argument("--select-target")
    selector.add_argument("--task-weights", default="")
    selector.add_argument("--semantic-weights", default="")
    selector.add_argument("--max-steps", type=int, default=5)
    selector.add_argument("--rollout-steps", type=int, default=5)
    selector.add_argument("--meta-dim", type=int, default=128)
    selector.add_argument("--hidden", type=int, default=512)
    selector.add_argument("--dropout", type=float, default=0.1)
    selector.add_argument("--gamma", type=float, default=0.9)
    selector.add_argument("--lambda-semantic-reward", type=float, default=0.2)
    selector.add_argument("--step-cost", type=float, default=0.03)
    selector.add_argument("--disable-stop", action="store_true")
    selector.add_argument("--epochs", type=int, default=60)
    selector.add_argument("--batch-size", type=int, default=16)
    selector.add_argument("--eval-batch-size", type=int, default=32)
    selector.add_argument("--lr", type=float, default=1e-4)
    selector.add_argument("--weight-decay", type=float, default=1e-4)
    selector.add_argument("--max-grad-norm", type=float, default=5.0)
    selector.set_defaults(function=train_selector)

    evaluate = subparsers.add_parser("evaluate")
    add_shared(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--missing", required=True)
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--policy", choices=("q", "random_one", "random_k"), default="q")
    evaluate.add_argument("--rollout-steps", type=int)
    evaluate.add_argument("--disable-stop", action="store_true")
    evaluate.add_argument("--eval-batch-size", type=int, default=32)
    evaluate.add_argument("--metrics-json")
    evaluate.add_argument("--variant", default="full")
    evaluate.set_defaults(function=evaluate_selector)

    inspect = subparsers.add_parser("inspect")
    add_shared(inspect)
    inspect.set_defaults(function=inspect_data)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()



