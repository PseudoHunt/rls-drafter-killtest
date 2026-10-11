"""Offline calibration trainer (rlsdk/calibrate.py) on the tiny model (CPU): the training forward is the S2 replay's
drafting forward; padding is inert; training lowers the loss."""
import json

import torch

import rlsdk.calibrate as cal
from rlsdk.adapters import FarSel
from rlsdk.replay import simulate, teacher_forced
from rlsdk.target_attn import default_spec
from test_replay import NEW, P, _generate, _setup

SMALL_W, SMALL_N = 40, 2


def _feats(monkeypatch):
    monkeypatch.setattr(cal, "W", SMALL_W)
    monkeypatch.setattr(cal, "N_FAR", SMALL_N)
    t, d, prompt = _setup()
    ids = _generate(t, d, prompt).output_ids[0]
    tf = teacher_forced(t, ids, d.target_layer_ids, attn_spec={"mid3": default_spec(t)["mid3"]})
    f = {"trace_id": "x", "P": P, "ids": ids, "ctx": tf.ctx, "hstar": tf.hstar, "score": tf.attn["mid3"].float()}
    return t, d, ids, tf, f


def test_training_forward_equals_s2_replay_drafting(monkeypatch):
    """For every replay round (frontier t) the teacher-forced S2-layout forward gives the replay's drafting z."""
    t, d, ids, tf, f = _feats(monkeypatch)
    seen = {}

    class Spy(FarSel):
        def apply(self, z, slots):
            seen[self._t] = z.detach().clone()
            return z

    ad = Spy({"W": SMALL_W, "keep": "prompt", "n": SMALL_N, "score": "mid3"})
    orig_begin = cal.WindowCache.begin

    def begin(self, c0, start, vs):  # record the frontier of the round being drafted
        ad._t = start
        return orig_begin(self, c0, start, vs)

    monkeypatch.setattr(cal.WindowCache, "begin", begin)
    simulate(d, t, ids, P, tf, ad, NEW)
    monkeypatch.setattr(cal.WindowCache, "begin", orig_begin)
    ts = [x for x in sorted(seen) if x <= ids.shape[0] - 16]
    assert len(ts) > 50
    far_active = 0
    for tt in ts:
        th, noise, pos, mask, _ = cal.build_batch(d, t, [f], [(0, tt)], ids.device)
        with torch.no_grad():
            z = cal.s2_forward(d, th, noise, pos, mask)[0]
        assert torch.allclose(z, seen[tt], atol=1e-5, rtol=1e-5), (tt, (z - seen[tt]).abs().max())
        far_active += cal.visible_rows(d, f, tt, ids.device).numel() > P + SMALL_W
    assert far_active > 20  # far chunks really selected in many rounds


def test_padded_batch_equals_singles(monkeypatch):
    t, d, ids, tf, f = _feats(monkeypatch)
    batch = [(0, P), (0, P + 30), (0, ids.shape[0] - 16), (0, P + 100)]
    th, noise, pos, mask, hs = cal.build_batch(d, t, [f], batch, ids.device)
    with torch.no_grad():
        zb = cal.s2_forward(d, th, noise, pos, mask)
        for b, a in enumerate(batch):
            z1 = cal.s2_forward(d, *cal.build_batch(d, t, [f], [a], ids.device)[:4])[0]
            assert torch.allclose(zb[b], z1, atol=1e-5, rtol=1e-5)
    assert torch.equal(hs[1], tf.hstar[P + 30:P + 45])


def test_anchor_range_and_determinism():
    feats = [{"P": 10, "ids": torch.zeros(60)}, {"P": 5, "ids": torch.zeros(30)}]
    a1, a2 = cal.anchors(feats, 500, 0), cal.anchors(feats, 500, 0)
    assert a1 == a2
    for i, tt in a1:
        assert feats[i]["P"] <= tt <= feats[i]["ids"].shape[0] - 16
    assert {i for i, _ in a1} == {0, 1}


def test_training_lowers_loss(monkeypatch):
    t, d, ids, tf, f = _feats(monkeypatch)
    pristine = {k: v.clone() for k, v in d.state_dict().items()}
    losses = cal.train(d, t, [f], 1e-3, 30, 8, ids.device, log=lambda m: None, pristine=pristine)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
    d.load_state_dict(pristine)


def test_loop_cut_applied(tmp_path):
    rows = [{"trace_id": "a", "prompt_len": 3, "prompt_ids": [1, 2, 3], "output_ids": list(range(2000)), "loop": False},
            {"trace_id": "b", "prompt_len": 3, "prompt_ids": [1, 2, 3],
             "output_ids": list(range(3000)) + list(range(5000, 5040)) * 100, "loop": True},
            {"trace_id": "c", "prompt_len": 3, "prompt_ids": [1, 2, 3],
             "output_ids": list(range(900)) + list(range(5000, 5040)) * 100, "loop": True}]
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = {r["trace_id"]: r for r in cal.load_cal_traces(str(p))}
    assert len(out["a"]["output_ids"]) == 2000
    assert 1024 <= len(out["b"]["output_ids"]) < 3000  # truncated before the loop onset
    assert "c" not in out  # < 1024 tokens kept -> dropped
