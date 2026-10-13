"""Average LoRA adapters of one backbone into full weights, without training (a "model soup").

Every adapter is a low-rank change of the same base weights, ``delta = scale * B @ A`` on each target
module, so the mean of N merged models is the base plus the mean of their deltas: that is what this
writes, in the layout ``scripts/lora_train.py`` merges into (``merged/``: the weights, then the original
repository's config, tokenizer and pooling files), so vLLM serves it the same way (compose profile
``lora``, port 8097). Uses: the checkpoints of one run (``--keep-all``'s ``checkpoints/step-N``), or the
adapters of runs that differ only in their seed. One adapter is merged as ``lora_train.py`` merges it, so
a scored model's ``merged/`` can be deleted and made again from its adapter. CPU is enough (the base
weights in bf16, ~16 GB of memory); ``[lora]`` extra. On the GB10, from ~/callbus:

    uv run --extra lora python scripts/lora_soup.py --adapter data/lora/qwen3-emb-lora-20k/adapter \\
      --adapter data/lora/qwen3-emb-lora-20k-r1/adapter --out data/lora/qwen3-emb-lora-soup-seeds

Each delta is computed in float32 and the mean added to the bf16 weight in float32, rounded once, as
peft's ``merge_and_unload`` adds one adapter's: a soup of one adapter is that adapter merged, bit for bit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

PREFIX = "base_model.model."  # peft's wrapper around the CausalLM in an adapter's tensor names


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--adapter", action="append", required=True, help="an adapter dir (repeat for a soup)")
    p.add_argument("--out", required=True, help="merged/ and soup.json go here")
    p.add_argument("--model", default=None, help="the base model (default: the adapters' own)")
    return p.parse_args(argv)


# -- pure parts (tested without a model) --------------------------------------------------------
def scale(cfg: dict[str, Any]) -> float:
    """peft's scaling of an adapter: alpha / r, or alpha / sqrt(r) under rsLoRA."""
    r, alpha = cfg["r"], cfg["lora_alpha"]
    return alpha / r**0.5 if cfg.get("use_rslora") else alpha / r


def check_configs(cfgs: list[dict[str, Any]]) -> str:
    """-> the base model all adapters were trained on. A soup averages changes of one base model, so
    adapters of different bases, or ones that train more than the LoRA matrices, are refused."""
    bases = {c.get("base_model_name_or_path") for c in cfgs}
    if len(bases) != 1:
        raise ValueError(f"the adapters change different base models: {sorted(map(str, bases))}")
    for c in cfgs:
        if c.get("peft_type", "LORA") != "LORA":
            raise ValueError(f"not a LoRA adapter: {c.get('peft_type')}")
        if c.get("modules_to_save") or c.get("bias", "none") != "none":
            raise ValueError("an adapter trains more than its LoRA matrices (modules_to_save or bias)")
        if c.get("fan_in_fan_out") or c.get("use_dora") or c.get("rank_pattern") or c.get("alpha_pattern"):
            raise ValueError("fan_in_fan_out, DoRA and per-module ranks or alphas are not handled")
    return bases.pop()


def module_pairs(tensors: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """An adapter's tensors -> {module name in the CausalLM: (A, B)}. Names look like
    ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight``; a name of another form, or a
    module with only one of its two matrices, is an error rather than a silent skip."""
    a_, b_ = {}, {}
    for name, t in tensors.items():
        if not name.startswith(PREFIX):
            raise ValueError(f"unexpected tensor name in an adapter: {name}")
        stem = name[len(PREFIX) :]
        for suffix, into in ((".lora_A.weight", a_), (".lora_B.weight", b_)):
            if stem.endswith(suffix):
                into[stem[: -len(suffix)]] = t
                break
        else:
            raise ValueError(f"not a LoRA matrix: {name}")
    if set(a_) != set(b_):
        raise ValueError(f"modules with one matrix only: {sorted(set(a_) ^ set(b_))[:3]}")
    return {m: (a_[m], b_[m]) for m in a_}


def mean_delta(parts: list[tuple[Any, Any, float]]) -> Any:
    """The mean of ``s * B @ A`` over the adapters that change one module, in float32."""
    total = None
    for a, b, s in parts:
        d = (b.float() @ a.float()) * s
        total = d if total is None else total + d
    return total / len(parts)


def main(argv: list[str] | None = None) -> int:
    a = parse(argv)
    import torch
    from lora_train import copy_original_files
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    dirs = [Path(d) for d in a.adapter]
    cfgs = [json.loads((d / "adapter_config.json").read_text()) for d in dirs]
    model = a.model or check_configs(cfgs)
    if a.model:
        check_configs(cfgs)
    adapters = [module_pairs(load_file(d / "adapter_model.safetensors")) for d in dirs]
    modules = set(adapters[0])
    if any(
        set(x) != modules for x in adapters[1:]
    ):  # a module missing from one adapter would be averaged with zero
        raise SystemExit("the adapters change different modules: train them with the same --rank and targets")
    out = Path(a.out)
    if (out / "merged").exists():
        raise SystemExit(f"{out / 'merged'} exists: pick another --out")
    print(f"{len(dirs)} adapters of {model}, {len(modules)} modules each", flush=True)

    base = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16)
    named = dict(base.named_modules())
    with torch.no_grad():
        for m in sorted(modules):
            w = named[m].weight
            delta = mean_delta([(x[m][0], x[m][1], scale(c)) for x, c in zip(adapters, cfgs, strict=True)])
            if delta.shape != w.shape:
                raise SystemExit(f"{m}: delta {tuple(delta.shape)} vs weight {tuple(w.shape)}")
            w += delta  # float32 into bf16: added in float32, rounded once, as peft merges
    base.save_pretrained(out / "merged", safe_serialization=True)
    copy_original_files(model, out / "merged")
    (out / "soup.json").write_text(
        json.dumps(
            {"model": model, "adapters": [str(d) for d in dirs], "scales": [scale(c) for c in cfgs]}, indent=2
        )
    )
    print(f"soup of {len(dirs)} in {out / 'merged'} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
