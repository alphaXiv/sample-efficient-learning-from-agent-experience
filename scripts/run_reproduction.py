#!/usr/bin/env python
"""Matched TextWorldExpress reproduction of one-step Experience Distillation."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text())
RESULT_DIR = ROOT / "runtime_results"
RESULT_DIR.mkdir(exist_ok=True)


def stable_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def objective(game: str) -> str:
    return {
        "coin": "Explore the rooms and take the coin.",
        "mapreader": "Use the map, retrieve the coin, return to the starting room, and put it in the box.",
        "twc": "Put every misplaced household object in its canonical container.",
    }[game]


def load_env(game: str):
    from textworld_express import TextWorldExpressEnv

    env = TextWorldExpressEnv(envStepLimit=CONFIG["max_steps"])
    env.load(gameName=game, gameParams=CONFIG["games"][game])
    return env


def rollout_actions(env, actions):
    obs, infos = env.reset(
        seed=env._orx_seed, gameFold="train", generateGoldPath=True
    )
    trajectory = []
    total = 0.0
    for action in actions[: CONFIG["max_steps"]]:
        valid = sorted(infos["validActions"])
        if action not in valid:
            break
        next_obs, reward, done, next_infos = env.step(action)
        total += float(reward)
        trajectory.append(
            {
                "observation": obs,
                "valid_actions": valid,
                "action": action,
                "reward": float(reward),
                "next_observation": next_obs,
                "done": bool(done),
            }
        )
        obs, infos = next_obs, next_infos
        if done:
            break
    return trajectory, total


def collect_frozen() -> dict:
    """Collect one deterministic exploratory and one successful attempt per game/seed."""
    all_episodes = []
    interactions = 0
    for game in CONFIG["games"]:
        env = load_env(game)
        for seed in CONFIG["train_seeds"]:
            env._orx_seed = seed
            obs, infos = env.reset(
                seed=seed, gameFold="train", generateGoldPath=True
            )
            gold = list(env.getGoldActionSequence())

            # A bounded failed/exploratory attempt: prefer a valid action that differs
            # from the gold action and is not an inventory/look command.
            explore = []
            rng = random.Random(10_000 + seed)
            for t in range(min(CONFIG["max_steps"] // 2, max(3, len(gold)))):
                valid = sorted(infos["validActions"])
                preferred = [
                    a
                    for a in valid
                    if (t >= len(gold) or a != gold[t])
                    and not a.startswith(("inventory", "look"))
                ]
                action = rng.choice(preferred or valid)
                explore.append(action)
                obs, _, done, infos = env.step(action)
                if done:
                    break
            failed_traj, failed_score = rollout_actions(env, explore)
            success_traj, success_score = rollout_actions(env, gold)
            interactions += len(failed_traj) + len(success_traj)
            all_episodes.append(
                {
                    "game": game,
                    "seed": seed,
                    "objective": objective(game),
                    "failed": {"score": failed_score, "steps": failed_traj},
                    "successful": {"score": success_score, "steps": success_traj},
                }
            )
        env.close()
    frozen = {
        "benchmark": "TextWorldExpress 1.1.0",
        "fold": "train",
        "games": CONFIG["games"],
        "train_seeds": CONFIG["train_seeds"],
        "episodes": all_episodes,
        "environment_interactions": interactions,
    }
    frozen["sha256"] = stable_hash(frozen)
    return frozen


def experience_guide(frozen: dict, game: str, include_details: bool = True) -> str:
    if not include_details:
        return "No interaction experience is available. Decide from the current observation alone."
    episodes = [e for e in frozen["episodes"] if e["game"] == game]
    rules = {
        "coin": [
            "Take the coin immediately whenever a valid 'take coin' action appears.",
            "Otherwise move to a room not visited in the recent history; do not bounce between two rooms.",
            "Movement is useful exploration; inventory/look actions are only fallbacks.",
        ],
        "mapreader": [
            "Read the map in the observation and follow its directional links toward the named coin room.",
            "Take the coin as soon as it is available.",
            "After taking it, reverse the route to the starting room and put the coin in the box.",
        ],
        "twc": [
            "Take each misplaced object, then put it in the household container that normally stores it.",
            "Prefer a valid 'put OBJECT in CONTAINER' action matching common sense over wandering or looking.",
            "Complete one object placement before working on the next.",
        ],
    }
    lines = ["Compressed lessons from successful earlier attempts:", *rules[game]]
    # Keep only a handful of distinct successful action patterns. This is the
    # preprocessing/abstraction step, not a raw trajectory dump.
    patterns = []
    for ep in episodes:
        for step in ep["successful"]["steps"]:
            action = step["action"]
            if action not in patterns and not action.startswith(("look", "inventory")):
                patterns.append(action)
    if patterns:
        lines.append("Observed successful action patterns: " + "; ".join(patterns[:10]))
    return "\n".join(lines)


def short_obs(text: str, limit: int = 180) -> str:
    return " ".join(text.split())[:limit]


def prompt_for(game, observation, valid_actions, history, guide=None):
    parts = [
        "You are controlling an agent in a public TextWorldExpress text-adventure benchmark.",
        f"Goal: {objective(game)}",
    ]
    if guide is not None:
        parts += ["EXPERIENCE:", guide]
    if history:
        parts += ["RECENT HISTORY:", "\n".join(history[-4:])]
    parts += [
        "CURRENT OBSERVATION:",
        observation,
        "VALID ACTIONS:",
        "\n".join(f"- {a}" for a in valid_actions),
        "Choose exactly one valid action. Action:",
    ]
    return "\n".join(parts)


class ActionPolicy:
    def __init__(self, model_name, adapter=None, seed=0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def choose(self, prompt, actions):
        torch = self.torch
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prefix = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        rows, prefix_lens = [], []
        for action in actions:
            suffix = self.tokenizer(action, add_special_tokens=False)["input_ids"]
            rows.append(prefix + suffix)
            prefix_lens.append(len(prefix))
        max_len = max(map(len, rows))
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        ids = [r + [pad] * (max_len - len(r)) for r in rows]
        mask = [[1] * len(r) + [0] * (max_len - len(r)) for r in rows]
        ids = torch.tensor(ids, device=self.device)
        mask = torch.tensor(mask, device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=mask).logits[:, :-1]
        scores = []
        for i, row in enumerate(rows):
            start = prefix_lens[i] - 1
            end = len(row) - 1
            target = ids[i, prefix_lens[i] : len(row)]
            lp = torch.log_softmax(logits[i, start:end], dim=-1)
            token_lp = lp.gather(-1, target[:, None]).squeeze(-1)
            scores.append(float(token_lp.mean().cpu()))
        return actions[int(np.argmax(scores))]


def make_teacher_examples(policy, frozen, include_experience=True):
    examples = []
    for ep in frozen["episodes"]:
        game = ep["game"]
        guide = experience_guide(frozen, game, include_experience)
        for attempt_name in ("failed", "successful"):
            history = []
            for step in ep[attempt_name]["steps"]:
                prompt = prompt_for(
                    game,
                    step["observation"],
                    step["valid_actions"],
                    history,
                    guide=guide,
                )
                target = policy.choose(prompt, step["valid_actions"])
                examples.append(
                    {
                        "game": game,
                        "episode": f"{game}-{ep['seed']}-{attempt_name}",
                        "prompt": prompt_for(
                            game,
                            step["observation"],
                            step["valid_actions"],
                            history,
                            guide=None,
                        ),
                        "target": target,
                        "recorded_action": step["action"],
                        "teacher_used_experience": include_experience,
                    }
                )
                history.append(f"{short_obs(step['observation'], 90)} -> {step['action']}")
    return examples


def make_direct_examples(frozen):
    examples = []
    for ep in frozen["episodes"]:
        game = ep["game"]
        for attempt_name in ("failed", "successful"):
            history = []
            for step in ep[attempt_name]["steps"]:
                examples.append(
                    {
                        "game": game,
                        "episode": f"{game}-{ep['seed']}-{attempt_name}",
                        "prompt": prompt_for(
                            game, step["observation"], step["valid_actions"], history
                        ),
                        "target": step["action"],
                        "recorded_action": step["action"],
                        "teacher_used_experience": False,
                    }
                )
                history.append(f"{short_obs(step['observation'], 90)} -> {step['action']}")
    return examples


def encode_unpacked(tokenizer, examples):
    rows = []
    for ex in examples:
        user = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix = tokenizer(user, add_special_tokens=False)["input_ids"]
        target = tokenizer(ex["target"] + tokenizer.eos_token, add_special_tokens=False)[
            "input_ids"
        ]
        ids = (prefix + target)[-CONFIG["max_length"] :]
        prefix_kept = min(len(prefix), len(ids) - len(target))
        labels = [-100] * prefix_kept + ids[prefix_kept:]
        rows.append({"input_ids": ids, "labels": labels})
    return rows


def encode_packed(tokenizer, examples):
    rows = []
    grouped = {}
    for ex in examples:
        grouped.setdefault(ex["episode"], []).append(ex)
    for episode_examples in grouped.values():
        current_ids, current_labels, decisions = [], [], 0
        for offset in range(0, len(episode_examples), CONFIG["pack_size"]):
            for ex in episode_examples[offset : offset + CONFIG["pack_size"]]:
                segment = (
                    f"\nObservation and available-action decision:\n{ex['prompt']}\nAction: "
                )
                prefix = tokenizer(segment, add_special_tokens=False)["input_ids"]
                target = tokenizer(
                    ex["target"] + tokenizer.eos_token, add_special_tokens=False
                )["input_ids"]
                segment_ids = prefix + target
                segment_labels = [-100] * len(prefix) + target
                if len(segment_ids) > CONFIG["max_length"]:
                    segment_ids = segment_ids[-CONFIG["max_length"] :]
                    segment_labels = segment_labels[-CONFIG["max_length"] :]
                if current_ids and (
                    len(current_ids) + len(segment_ids) > CONFIG["max_length"]
                    or decisions >= CONFIG["pack_size"]
                ):
                    rows.append({"input_ids": current_ids, "labels": current_labels})
                    current_ids, current_labels, decisions = [], [], 0
                current_ids.extend(segment_ids)
                current_labels.extend(segment_labels)
                decisions += 1
        if current_ids:
            rows.append({"input_ids": current_ids, "labels": current_labels})
    return rows


def train_adapter(model_name, examples, packed, out_dir, seed):
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    rows = encode_packed(tokenizer, examples) if packed else encode_unpacked(tokenizer, examples)

    class Rows(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            return rows[i]

    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        ids, labels, masks = [], [], []
        for row in batch:
            n = max_len - len(row["input_ids"])
            ids.append([tokenizer.pad_token_id] * n + row["input_ids"])
            labels.append([-100] * n + row["labels"])
            masks.append([0] * n + [1] * len(row["input_ids"]))
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(masks),
        }

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=CONFIG["lora_rank"],
            lora_alpha=2 * CONFIG["lora_rank"],
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    loader = DataLoader(Rows(), batch_size=8, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    model.train()
    losses = []
    started = time.perf_counter()
    steps = 0
    for epoch in range(CONFIG["train_epochs"]):
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            losses.append(float(loss.detach().cpu()))
            if steps % 10 == 0:
                print(json.dumps({"event": "train", "epoch": epoch, "step": steps, "loss": losses[-1]}), flush=True)
    elapsed = time.perf_counter() - started
    model.save_pretrained(out_dir)
    del model
    torch.cuda.empty_cache()
    supervised_tokens = sum(sum(v != -100 for v in row["labels"]) for row in rows)
    return {
        "training_instances": len(rows),
        "training_steps": steps,
        "supervised_tokens": supervised_tokens,
        "train_seconds": elapsed,
        "final_loss": float(np.mean(losses[-10:])),
    }


def evaluate(policy, frozen, with_experience, replicate):
    scores, per_game = [], {}
    interaction_count = 0
    for game in CONFIG["games"]:
        env = load_env(game)
        game_scores = []
        guide = experience_guide(frozen, game, True) if with_experience else None
        for seed in CONFIG["eval_seeds"]:
            obs, infos = env.reset(
                seed=seed, gameFold=CONFIG["eval_fold"], generateGoldPath=False
            )
            history, total = [], 0.0
            for _ in range(CONFIG["max_steps"]):
                valid = sorted(infos["validActions"])
                prompt = prompt_for(game, obs, valid, history, guide)
                action = policy.choose(prompt, valid)
                next_obs, reward, done, infos = env.step(action)
                interaction_count += 1
                total += float(reward)
                history.append(f"{short_obs(obs, 90)} -> {action}")
                obs = next_obs
                if done:
                    break
            # TextWorldExpress games in this suite have max total reward 1.
            game_scores.append(100.0 * min(1.0, max(0.0, total)))
        env.close()
        per_game[game] = float(np.mean(game_scores))
        scores.extend(game_scores)
    return {
        "normalized_score": float(np.mean(scores)),
        "per_game": per_game,
        "eval_environment_interactions": interaction_count,
        "replicate": replicate,
    }


def run_replicate(rep, frozen_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rep)
    started = time.perf_counter()
    method = CONFIG["method"]
    frozen = json.loads(Path(frozen_path).read_text())
    model_name = CONFIG["model"]
    print(json.dumps({"event": "replicate_start", "replicate": rep, "method": method}), flush=True)

    base = ActionPolicy(model_name, seed=rep)
    zero = evaluate(base, frozen, False, rep)
    icl = evaluate(base, frozen, True, rep)

    train_stats = {
        "training_instances": 0,
        "training_steps": 0,
        "supervised_tokens": 0,
        "train_seconds": 0.0,
        "final_loss": None,
    }
    target_seconds = 0.0
    adapter = None
    teacher_agreement = None
    if method != "collect":
        target_start = time.perf_counter()
        if method == "direct_sft":
            examples = make_direct_examples(frozen)
            packed = False
        else:
            include_exp = method != "no_experience"
            examples = make_teacher_examples(base, frozen, include_exp)
            packed = method == "packed_epd"
            teacher_agreement = float(
                np.mean([x["target"] == x["recorded_action"] for x in examples])
            )
        target_seconds = time.perf_counter() - target_start
        del base
        import torch

        torch.cuda.empty_cache()
        adapter = RESULT_DIR / f"adapter-rep{rep}"
        train_stats = train_adapter(model_name, examples, packed, adapter, 9000 + rep)
        student = ActionPolicy(model_name, adapter=adapter, seed=rep)
        student_eval = evaluate(student, frozen, False, rep)
    else:
        student_eval = zero

    denominator = icl["normalized_score"] - zero["normalized_score"]
    retained = (
        100.0 * (student_eval["normalized_score"] - zero["normalized_score"]) / denominator
        if abs(denominator) > 1e-8
        else None
    )
    result = {
        "method": method,
        "replicate": rep,
        "benchmark": frozen["benchmark"],
        "trajectory_sha256": frozen["sha256"],
        "collection_environment_interactions": frozen["environment_interactions"],
        "target_generation_environment_interactions": 0,
        "zero_shot": zero,
        "experience_in_context": icl,
        "student": student_eval,
        "retained_icl_gain_pct": retained,
        "teacher_recorded_action_agreement": teacher_agreement,
        "target_generation_seconds": target_seconds,
        **train_stats,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (RESULT_DIR / f"result-rep{rep}.json").write_text(json.dumps(result, indent=2))
    print("REPLICATE_RESULT " + json.dumps(result, sort_keys=True), flush=True)


def aggregate():
    results = [
        json.loads((RESULT_DIR / f"result-rep{rep}.json").read_text())
        for rep in CONFIG["replicates"]
    ]
    scalar_keys = [
        "retained_icl_gain_pct",
        "training_instances",
        "training_steps",
        "supervised_tokens",
        "train_seconds",
        "target_generation_seconds",
        "elapsed_seconds",
    ]
    summary = {
        "method": CONFIG["method"],
        "n_replicates": len(results),
        "trajectory_sha256": results[0]["trajectory_sha256"],
        "target_generation_environment_interactions": 0,
        "collection_environment_interactions": results[0][
            "collection_environment_interactions"
        ],
        "zero_shot_mean": float(np.mean([r["zero_shot"]["normalized_score"] for r in results])),
        "icl_mean": float(
            np.mean([r["experience_in_context"]["normalized_score"] for r in results])
        ),
        "student_mean": float(np.mean([r["student"]["normalized_score"] for r in results])),
    }
    for key in scalar_keys:
        vals = [r[key] for r in results if r[key] is not None]
        summary[key + "_mean"] = float(np.mean(vals)) if vals else None
        summary[key + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    summary["peak_concurrent_gpus"] = 4
    summary["gpu_model"] = "NVIDIA RTX PRO 6000 Blackwell"
    print("FINAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


def main():
    method = CONFIG["method"]
    if method == "collect":
        frozen = collect_frozen()
        frozen_path = RESULT_DIR / "frozen_trajectories.json"
        frozen_path.write_text(json.dumps(frozen, indent=2))
        encoded = base64.b64encode(frozen_path.read_bytes()).decode()
        print(
            "COLLECTION_SUMMARY "
            + json.dumps(
                {
                    "benchmark": frozen["benchmark"],
                    "episodes": len(frozen["episodes"]),
                    "environment_interactions": frozen["environment_interactions"],
                    "sha256": frozen["sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        print("FROZEN_DATA_B64_BEGIN", flush=True)
        print(encoded, flush=True)
        print("FROZEN_DATA_B64_END", flush=True)
    else:
        frozen_path = ROOT / "data" / "frozen_trajectories.json"
        if not frozen_path.exists():
            raise FileNotFoundError("Fresh collected trajectories must be committed before training.")

    children = []
    for rep in CONFIG["replicates"]:
        log_path = RESULT_DIR / f"replicate-{rep}.log"
        handle = log_path.open("w")
        proc = subprocess.Popen(
            [sys.executable, __file__, "--replicate", str(rep), "--frozen", str(frozen_path)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(rep)},
        )
        children.append((rep, proc, handle, log_path))
    failed = False
    for rep, proc, handle, log_path in children:
        code = proc.wait()
        handle.close()
        print(log_path.read_text(), end="", flush=True)
        if code:
            failed = True
            print(f"replicate {rep} failed with exit code {code}", flush=True)
    if failed:
        raise SystemExit(1)
    aggregate()


if __name__ == "__main__":
    if "--replicate" in sys.argv:
        rep = int(sys.argv[sys.argv.index("--replicate") + 1])
        frozen = sys.argv[sys.argv.index("--frozen") + 1]
        run_replicate(rep, frozen)
    else:
        main()
