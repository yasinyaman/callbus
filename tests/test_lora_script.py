"""The pure parts of scripts/lora_train.py: the InfoNCE masking (torch), the step plan and the
learning-rate schedule. The model itself is never loaded here."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("lora_train", ROOT / "scripts" / "lora_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plan_and_schedule():
    m = _script()
    assert m.plan_steps(100, 16, 2, 1.0) == (7, 3) and m.plan_steps(64, 16, 2, 0.5) == (2, 1)
    assert m.lr_at(0, 100, 10, 1e-4) == pytest.approx(1e-5) and m.lr_at(9, 100, 10, 1e-4) == pytest.approx(
        1e-4
    )
    assert m.lr_at(10, 100, 10, 1e-4) == pytest.approx(1e-4)
    assert m.lr_at(100, 100, 10, 1e-4) == pytest.approx(0.0, abs=1e-12)
    assert math.isclose(m.lr_at(55, 100, 10, 1e-4), 0.5e-4, rel_tol=1e-6)  # halfway down the cosine


def test_infonce_masks_duplicate_positives():
    torch = pytest.importorskip("torch")
    m = _script()
    sim = torch.tensor([[0.9, 0.9, 0.1], [0.9, 0.9, 0.1], [0.0, 0.0, 0.8]])
    plain = m.infonce(sim, ["a", "b", "c"], tau=0.1)
    masked = m.infonce(sim, ["a", "a", "c"], tau=0.1)  # rows 0 and 1 share a positive text
    assert masked < plain  # the duplicate is no longer a competing negative
    assert torch.isfinite(masked)


def test_the_false_negative_margin_masks_negatives_scored_above_the_positive():
    torch = pytest.importorskip("torch")
    m = _script()
    # row 0's positive scores 0.5, a negative 0.7: above it by 0.2, likely the same tool written twice
    sim = torch.tensor([[0.5, 0.7, 0.1], [0.2, 0.9, 0.1], [0.0, 0.1, 0.8]])
    texts = ["a", "b", "c"]
    plain = m.infonce(sim, texts, tau=0.1)
    assert m.infonce(sim, texts, tau=0.1, fn_margin=0.1) < plain  # masked: no longer pulls row 0 down
    assert torch.equal(m.infonce(sim, texts, tau=0.1, fn_margin=0.3), plain)  # 0.2 is within 0.3
    grad = sim.clone().requires_grad_(True)
    m.infonce(grad, texts, tau=0.1, fn_margin=0.1).backward()
    assert grad.grad[0, 1] == 0 and grad.grad[0, 2] != 0  # the masked pair gets no gradient


def test_shapes_and_shape_balanced_draws():
    m = _script()
    assert m.doc_shape('{"name": "a", "description": "b", "parameters": {}}') == "description,name,parameters"
    assert m.doc_shape("{'name': 'a', 'api_call': 'f()'}") == "api_call,name"  # Python repr, as ToolRet has
    assert m.doc_shape("plain words") == "<text>" and m.doc_shape("[1, 2]") == "<list>"

    class P:
        def __init__(self, i, shape):
            self.id, self.positives = i, [json.dumps(dict.fromkeys(shape, 1))]

    pairs = [P(f"a{i}", "xy") for i in range(100)] + [P(f"b{i}", "xz") for i in range(50)]
    pairs += [P(f"c{i}", "q") for i in range(5)]
    got = m.balanced(pairs, 60, seed=0)
    by = {s: sum(m.doc_shape(p.positives[0]) == s for p in got) for s in ("x,y", "x,z", "q")}
    assert len(got) == 60 and len({p.id for p in got}) == 60
    assert by == {"q": 5, "x,y": 28, "x,z": 27}  # the small shape whole, the rest split evenly
    assert [p.id for p in m.balanced(pairs, 60, seed=0)] == [p.id for p in got]  # seeded
    assert [p.id for p in m.balanced(pairs, 60, seed=1)] != [p.id for p in got]
    assert len(m.balanced(pairs, 0, seed=0)) == 155 == len(m.balanced(pairs, 999, seed=0))


def test_the_data_seed_is_apart_from_the_training_seed():
    """A replica run changes --seed (initialisation, dropout, batch order) and keeps the pairs."""
    m = _script()
    base = ["--pairs", "p.jsonl", "--dev", "d", "--out", "o"]
    assert m.parse([*base, "--seed", "3"]).data_seed == 3  # default: --seed, as before
    a = m.parse([*base, "--seed", "3", "--data-seed", "0"])
    assert (a.seed, a.data_seed) == (3, 0)


def test_only_the_listed_files_of_a_full_snapshot_are_copied(tmp_path):
    """A cache that also holds the base weights must not put them next to the merged ones."""
    m = _script()
    src, merged = tmp_path / "snapshot", tmp_path / "merged"
    names = [
        "config.json", "tokenizer.json", "vocab.json", "1_Pooling/config.json", "modules.json",
        "model-00001-of-00004.safetensors", "model.safetensors.index.json", "README.md", "LICENSE",
    ]  # fmt: skip
    for name in names:
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        (src / name).write_text(name)
    merged.mkdir()
    (merged / "model.safetensors").write_text("merged weights")
    copied = m.copy_listed(src, merged)
    assert copied == ["1_Pooling/config.json", "config.json", "modules.json", "tokenizer.json", "vocab.json"]
    assert sorted(p.relative_to(merged).as_posix() for p in merged.rglob("*") if p.is_file()) == sorted(
        [*copied, "model.safetensors"]
    )


def _soup():
    spec = importlib.util.spec_from_file_location("lora_soup", ROOT / "scripts" / "lora_soup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_soup_takes_adapters_of_one_plain_lora_base():
    m = _soup()
    with pytest.raises(SystemExit):
        m.parse(["--out", "o"])
    assert m.parse(["--adapter", "a", "--adapter", "b", "--out", "o"]).adapter == ["a", "b"]
    cfg = {
        "base_model_name_or_path": "Qwen/Qwen3-Embedding-8B",
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
    }
    assert m.check_configs([cfg, dict(cfg)]) == "Qwen/Qwen3-Embedding-8B"
    assert m.scale(cfg) == 2.0 and m.scale({**cfg, "use_rslora": True}) == pytest.approx(8.0)
    for bad in (
        {**cfg, "base_model_name_or_path": "other"},
        {**cfg, "use_dora": True},
        {**cfg, "modules_to_save": ["lm_head"]},
        {**cfg, "rank_pattern": {"q_proj": 8}},
    ):
        with pytest.raises(ValueError):
            m.check_configs([cfg, bad])


def test_adapter_tensors_map_to_the_causal_lm_modules():
    m = _soup()
    stem = "base_model.model.model.layers.0.self_attn.q_proj"
    pairs = m.module_pairs({f"{stem}.lora_A.weight": "A", f"{stem}.lora_B.weight": "B"})
    assert pairs == {"model.layers.0.self_attn.q_proj": ("A", "B")}
    with pytest.raises(ValueError):  # B without its A
        m.module_pairs({f"{stem}.lora_B.weight": "B"})
    with pytest.raises(ValueError):  # not a LoRA matrix
        m.module_pairs({f"{stem}.weight": "W"})


def test_the_mean_delta_is_the_mean_of_the_merged_weights():
    torch = pytest.importorskip("torch")
    m = _soup()
    g = torch.Generator().manual_seed(0)
    w = torch.randn(6, 5, generator=g)
    adapters = [(torch.randn(2, 5, generator=g), torch.randn(6, 2, generator=g), s) for s in (2.0, 0.5, 1.0)]
    merged = [w + s * b @ a for a, b, s in adapters]  # what merge_and_unload gives per adapter
    assert torch.allclose(w + m.mean_delta(adapters), sum(merged) / 3, atol=1e-6)
    a, b, s = adapters[0]
    assert torch.allclose(m.mean_delta([adapters[0]]), s * b @ a)  # a soup of one = that adapter merged
    bf = [(x.bfloat16(), y.bfloat16(), z) for x, y, z in adapters]
    assert m.mean_delta(bf).dtype == torch.float32  # computed in float32 whatever the adapters hold


def test_grad_cache_gives_the_whole_batch_gradient_with_dropout():
    torch = pytest.importorskip("torch")
    m = _script()

    class Tiny(torch.nn.Module):  # two towers sharing a layer, with dropout like the LoRA layers
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(50, 8)
            self.mix = torch.nn.Linear(8, 8)
            self.drop = torch.nn.Dropout(0.3)

        def forward(self, ids):
            return torch.nn.functional.normalize(self.mix(self.drop(self.emb(ids))), dim=-1)

    def run(cached):
        torch.manual_seed(0)
        net = Tiny().train()
        q_ids, d_ids = torch.arange(12), torch.arange(12, 24)
        chunks = [(q_ids[i : i + 4], d_ids[i : i + 4]) for i in range(0, 12, 4)]
        texts = [str(i) for i in range(12)]
        texts[5] = texts[1]  # a duplicate positive: masked the same way in both

        def loss_fn(zq, zd):
            return m.infonce(zq @ zd.T, texts, tau=0.05, fn_margin=0.1)

        def encode(c, grad):
            with torch.set_grad_enabled(grad):
                return net(c[0]), net(c[1])

        if cached:
            loss = m.cached_step(encode, chunks, loss_fn)
        else:  # one graph over the whole batch, chunks drawn in the same order
            zs = [encode(c, True) for c in chunks]
            loss_t = loss_fn(torch.cat([z[0] for z in zs]), torch.cat([z[1] for z in zs]))
            loss_t.backward()
            loss = float(loss_t.detach())
        return loss, {n: p.grad.clone() for n, p in net.named_parameters()}, torch.rand(1)

    plain_loss, plain_grads, plain_next = run(cached=False)
    cached_loss, cached_grads, cached_next = run(cached=True)
    assert cached_loss == pytest.approx(plain_loss, rel=1e-6)
    for name, g in plain_grads.items():
        assert torch.allclose(cached_grads[name], g, atol=1e-6), name
    assert torch.equal(cached_next, plain_next)  # the RNG goes on where a plain step would leave it


def test_cache_batch_is_the_whole_step():
    m = _script()
    base = ["--pairs", "p.jsonl", "--dev", "d", "--out", "o"]
    assert (m.parse(base).cache_batch, m.parse(base).accumulate) == (0, 2)  # as before
    a = m.parse([*base, "--cache-batch", "128"])
    assert (a.cache_batch, a.accumulate, a.micro_batch) == (128, 1, 16)
    # --cache-batch 32 takes the replica's 625 steps of 32 pairs, with 31 negatives a request instead of 15
    assert m.plan_steps(20000, 32, 1, 1.0)[1] == m.plan_steps(20000, 16, 2, 1.0)[1] == 625
    with pytest.raises(SystemExit):  # GradCache accumulates nothing: one batch is one step
        m.parse([*base, "--cache-batch", "128", "--accumulate", "2"])
    with pytest.raises(SystemExit):
        m.parse([*base, "--cache-batch", "16"])  # no bigger than one micro-batch
