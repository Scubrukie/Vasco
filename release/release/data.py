from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


CONDITION_ORDER = ("A", "V", "T", "AV", "AT", "VT", "AVT")
CONDITION_MASKS: dict[str, tuple[int, int, int]] = {
    "A": (1, 0, 0),
    "V": (0, 1, 0),
    "T": (0, 0, 1),
    "AV": (1, 1, 0),
    "AT": (1, 0, 1),
    "VT": (0, 1, 1),
    "AVT": (1, 1, 1),
}
MISSING_ORDER = (
    "audio",
    "video",
    "text",
    "audio_video",
    "audio_text",
    "video_text",
)
MISSING_TO_INDEX = {name: index for index, name in enumerate(MISSING_ORDER)}
MISSING_TO_CONDITION = {
    "audio": "VT",
    "video": "AT",
    "text": "AV",
    "audio_video": "T",
    "audio_text": "V",
    "video_text": "A",
}


def torch_load(
    path: str | Path,
    *,
    trusted_checkpoint: bool = False,
) -> dict[str, Any]:
    source = Path(path)
    try:
        value = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(source, map_location="cpu")
    except pickle.UnpicklingError:
        if not trusted_checkpoint:
            raise
        value = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a dictionary PT package")
    return value


@dataclass(frozen=True)
class SequenceSlice:
    video_id: str
    sample_ids: list[str]
    audio: torch.Tensor
    vision: torch.Tensor
    text: torch.Tensor
    labels: torch.Tensor


class PackedData:
    """Read-only view of a normalized TSA-HC release PT package."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.payload = torch_load(self.path)
        meta = self.payload.get("meta", {})
        if meta.get("format") != "tsa_hc_release" or int(meta.get("version", 0)) != 2:
            raise ValueError(f"{self.path}: unsupported release PT format")
        if tuple(meta.get("missing_order", ())) != MISSING_ORDER:
            raise ValueError(f"{self.path}: missing-modality order does not match the release code")
        self.meta = meta
        self.features = self.payload["features"]
        self.actions = self.payload["actions"]
        self.action_embeddings = self.payload["action_vocab"]["embeddings"].float()
        if self.action_embeddings.ndim != 2:
            raise ValueError("action vocabulary embeddings must have shape [V,D]")
        self._validate()

    @property
    def dataset(self) -> str:
        return str(self.meta["dataset"])

    @property
    def action_dim(self) -> int:
        return int(self.action_embeddings.size(1))

    def feature_split(self, split: str) -> dict[str, Any]:
        if split not in self.features:
            raise KeyError(f"feature split {split!r} is unavailable")
        return self.features[split]

    def action_split(self, split: str) -> dict[str, Any]:
        if split not in self.actions:
            raise KeyError(f"action split {split!r} is unavailable")
        return self.actions[split]

    def sequence_dataset(self, split: str) -> "SequenceDataset":
        return SequenceDataset(self.feature_split(split))

    def projector_dataset(
        self,
        split: str,
        missing: Iterable[str],
    ) -> "ProjectorSequenceDataset":
        return ProjectorSequenceDataset(self, split, tuple(missing))

    def selector_dataset(
        self,
        split: str,
        missing: Iterable[str],
    ) -> "SelectorSequenceDataset":
        return SelectorSequenceDataset(self, split, tuple(missing))

    def _validate(self) -> None:
        vocabulary_size = int(self.action_embeddings.size(0))
        for split, features in self.features.items():
            ids = features["ids"]
            count = len(ids)
            if len(set(ids)) != count:
                raise ValueError(f"features/{split} contains duplicate sample IDs")
            if any(features[name].size(0) != count for name in ("audio", "vision", "text", "labels")):
                raise ValueError(f"features/{split} tensors have inconsistent first dimensions")
            offsets = features["sequence_offsets"].long()
            if offsets.ndim != 1 or int(offsets[0]) != 0 or int(offsets[-1]) != count:
                raise ValueError(f"features/{split} has invalid sequence offsets")
            if bool((offsets[1:] <= offsets[:-1]).any().item()):
                raise ValueError(f"features/{split} contains an empty sequence")
            if len(features["video_ids"]) + 1 != offsets.numel():
                raise ValueError(f"features/{split} video IDs do not match sequence offsets")

        for split, conditions in self.actions.items():
            valid_ids = set(self.feature_split(split)["ids"])
            for missing, group in conditions.items():
                if missing not in MISSING_TO_INDEX:
                    raise ValueError(f"actions/{split} has unknown condition {missing!r}")
                ids = group["ids"]
                action_ids = group["action_ids"]
                anchors = group["anchors"]
                if len(ids) != action_ids.size(0) or anchors.shape != action_ids.shape:
                    raise ValueError(f"actions/{split}/{missing} tensors are misaligned")
                if len(set(ids)) != len(ids):
                    raise ValueError(f"actions/{split}/{missing} contains duplicate IDs")
                unknown = set(ids) - valid_ids
                if unknown:
                    raise ValueError(
                        f"actions/{split}/{missing} references unknown sample {next(iter(unknown))!r}"
                    )
                if action_ids.numel():
                    low = int(action_ids.min())
                    high = int(action_ids.max())
                    if low < 0 or high >= vocabulary_size:
                        raise ValueError(
                            f"actions/{split}/{missing} action index range [{low},{high}] "
                            f"exceeds vocabulary size {vocabulary_size}"
                        )


class SequenceDataset(Dataset[SequenceSlice]):
    def __init__(self, split: dict[str, Any]):
        self.split = split
        self.offsets = split["sequence_offsets"].long()

    def __len__(self) -> int:
        return self.offsets.numel() - 1

    def __getitem__(self, index: int) -> SequenceSlice:
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return SequenceSlice(
            video_id=str(self.split["video_ids"][index]),
            sample_ids=list(self.split["ids"][start:end]),
            audio=self.split["audio"][start:end],
            vision=self.split["vision"][start:end],
            text=self.split["text"][start:end],
            labels=self.split["labels"][start:end],
        )


def collate_sequences(batch: list[SequenceSlice]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(batch)
    max_length = max(len(item.sample_ids) for item in batch)
    audio = torch.zeros(batch_size, max_length, 512)
    vision = torch.zeros(batch_size, max_length, 1024)
    text = torch.zeros(batch_size, max_length, 1024)
    labels = torch.zeros(batch_size, max_length)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    for row, item in enumerate(batch):
        length = len(item.sample_ids)
        audio[row, :length] = item.audio.float()
        vision[row, :length] = item.vision.float()
        text[row, :length] = item.text.float()
        labels[row, :length] = item.labels.float()
        mask[row, :length] = True
    return {
        "audio": audio,
        "vision": vision,
        "text": text,
        "labels": labels,
        "mask": mask,
        "video_ids": [item.video_id for item in batch],
        "sample_ids": [item.sample_ids for item in batch],
    }


def _action_maps(
    package: PackedData,
    split: str,
    missing_names: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    conditions = package.action_split(split)
    output: dict[str, dict[str, int]] = {}
    for missing in missing_names:
        if missing not in MISSING_TO_INDEX:
            raise ValueError(f"unknown missing modality: {missing!r}")
        if missing not in conditions:
            raise KeyError(f"release PT has no actions for {split}/{missing}")
        ids = conditions[missing]["ids"]
        output[missing] = {sample_id: row for row, sample_id in enumerate(ids)}
    return output


class ProjectorSequenceDataset(Dataset[dict[str, Any]]):
    """One item per video, missing condition, and candidate branch."""

    def __init__(self, package: PackedData, split: str, missing_names: tuple[str, ...]):
        self.package = package
        self.split_name = split
        self.sequences = package.sequence_dataset(split)
        self.conditions = package.action_split(split)
        self.row_maps = _action_maps(package, split, missing_names)
        self.items: list[tuple[int, str, int]] = []
        for sequence_index in range(len(self.sequences)):
            sample_ids = self.sequences[sequence_index].sample_ids
            for missing in missing_names:
                row_map = self.row_maps[missing]
                candidate_count = int(self.conditions[missing]["action_ids"].size(1))
                if any(sample_id in row_map for sample_id in sample_ids):
                    for branch in range(candidate_count):
                        self.items.append((sequence_index, missing, branch))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence_index, missing, branch = self.items[index]
        return {
            "sequence": self.sequences[sequence_index],
            "missing": missing,
            "missing_idx": MISSING_TO_INDEX[missing],
            "branch": branch,
            "rows": self.row_maps[missing],
            "condition": self.conditions[missing],
        }


def make_projector_collate(package: PackedData):
    vocabulary = package.action_embeddings

    def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        base = collate_sequences([item["sequence"] for item in items])
        batch_size, max_length = base["mask"].shape
        action = torch.zeros(batch_size, max_length, package.action_dim)
        anchors = torch.zeros(batch_size, max_length, dtype=torch.long)
        action_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
        missing_idx = torch.tensor([item["missing_idx"] for item in items], dtype=torch.long)
        for row, item in enumerate(items):
            branch = int(item["branch"])
            action_ids = item["condition"]["action_ids"]
            anchor_ids = item["condition"]["anchors"]
            row_map = item["rows"]
            for column, sample_id in enumerate(item["sequence"].sample_ids):
                group_row = row_map.get(sample_id)
                if group_row is None:
                    continue
                action[row, column] = vocabulary[int(action_ids[group_row, branch])]
                anchors[row, column] = int(anchor_ids[group_row, branch])
                action_mask[row, column] = True
        base.update(
            {
                "missing_idx": missing_idx,
                "action": action,
                "anchors": anchors,
                "action_mask": action_mask,
            }
        )
        return base

    return collate


class SelectorSequenceDataset(Dataset[dict[str, Any]]):
    """One item per video and missing condition with all candidates."""

    def __init__(self, package: PackedData, split: str, missing_names: tuple[str, ...]):
        self.package = package
        self.split_name = split
        self.sequences = package.sequence_dataset(split)
        self.conditions = package.action_split(split)
        self.row_maps = _action_maps(package, split, missing_names)
        self.items: list[tuple[int, str]] = []
        for sequence_index in range(len(self.sequences)):
            sample_ids = self.sequences[sequence_index].sample_ids
            for missing in missing_names:
                if any(sample_id in self.row_maps[missing] for sample_id in sample_ids):
                    self.items.append((sequence_index, missing))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence_index, missing = self.items[index]
        return {
            "sequence": self.sequences[sequence_index],
            "missing": missing,
            "missing_idx": MISSING_TO_INDEX[missing],
            "rows": self.row_maps[missing],
            "condition": self.conditions[missing],
        }


def make_selector_collate(package: PackedData):
    vocabulary = package.action_embeddings

    def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        base = collate_sequences([item["sequence"] for item in items])
        batch_size, max_length = base["mask"].shape
        candidate_counts = {int(item["condition"]["action_ids"].size(1)) for item in items}
        if len(candidate_counts) != 1:
            raise ValueError("a selector batch cannot mix different candidate counts")
        candidate_count = candidate_counts.pop()
        actions = torch.zeros(batch_size, max_length, candidate_count, package.action_dim)
        anchors = torch.zeros(batch_size, max_length, candidate_count, dtype=torch.long)
        candidate_mask = torch.zeros(batch_size, max_length, candidate_count, dtype=torch.bool)
        action_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
        missing_idx = torch.tensor([item["missing_idx"] for item in items], dtype=torch.long)
        for row, item in enumerate(items):
            action_ids = item["condition"]["action_ids"]
            anchor_ids = item["condition"]["anchors"]
            row_map = item["rows"]
            for column, sample_id in enumerate(item["sequence"].sample_ids):
                group_row = row_map.get(sample_id)
                if group_row is None:
                    continue
                ids = action_ids[group_row].long()
                actions[row, column] = vocabulary[ids]
                anchors[row, column] = anchor_ids[group_row].long()
                candidate_mask[row, column] = True
                action_mask[row, column] = True
        base.update(
            {
                "missing_idx": missing_idx,
                "actions": actions,
                "anchors": anchors,
                "candidate_mask": candidate_mask,
                "action_mask": action_mask,
            }
        )
        return base

    return collate


def observed_mask(missing_idx: torch.Tensor, length: int) -> torch.Tensor:
    table = missing_idx.new_tensor(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]
    )
    values = table[missing_idx.long()]
    return values[:, None, :].expand(-1, length, -1)

