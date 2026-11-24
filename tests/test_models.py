"""CNN architecture zoo: shapes, registry, training through the real trainer,
and loading through the predictor (including checkpoints from before the
registry existed)."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from idefect.config import Config  # noqa: E402
from idefect.models.custom_cnn import CustomCNN, count_parameters  # noqa: E402
from idefect.models.registry import available, build_cnn, display_name  # noqa: E402


@pytest.mark.parametrize("arch", available())
@pytest.mark.parametrize("size", [32, 128])
def test_every_architecture_maps_images_to_class_logits(arch, size):
    model = build_cnn(arch, num_classes=6).eval()
    with torch.no_grad():
        out = model(torch.randn(3, 1, size, size))
    assert out.shape == (3, 6)
    assert torch.isfinite(out).all()
    assert count_parameters(model) > 0
    assert display_name(arch) != arch or arch not in available()


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError):
        build_cnn("transformer", num_classes=6)


@pytest.fixture
def tiny(dummy_dataset, tmp_path) -> Config:
    c = Config()
    c.data.root = str(dummy_dataset)
    c.data.manifest = ""
    c.data.image_size = 32
    c.data.train_frac, c.data.val_frac, c.data.test_frac = 0.5, 0.25, 0.25
    c.cnn.epochs = 1
    c.cnn.batch_size = 8
    c.cnn.num_workers = 0
    c.cnn.device = "cpu"
    c.cnn.seeds = [0]
    c.data.duplicates.cache_dir = str(tmp_path / "cache")
    c.data.duplicates.review_file = ""
    c.data.duplicates.top_k = 3
    c.output.results_dir = str(tmp_path / "results")
    c.output.figures_dir = str(tmp_path / "figures")
    c.output.logs_dir = str(tmp_path / "logs")
    return c


@pytest.mark.parametrize("arch", ["vgg", "resnet", "densenet", "mobilenet"])
def test_architecture_trains_and_reloads(tiny, arch, dummy_dataset):
    from idefect.inference.predict import Predictor
    from idefect.training.train_cnn import train_cnn

    tiny.cnn.arch = arch
    metrics = train_cnn(tiny)
    results = Path(tiny.output.results_dir)
    assert metrics["arch"] == arch and metrics["model"] == arch
    assert metrics["display_name"] == display_name(arch)
    saved = json.loads((results / f"{arch}_metrics.json").read_text())
    assert saved["seed_summary"]["seeds"] == [0]
    assert (results / "cnn_runs" / arch / "seed0" / f"{arch}.pt").exists()
    pred = Predictor(results / f"{arch}.pt").predict(next(Path(dummy_dataset).rglob("*.png")))
    assert pred.model_kind == arch
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-5


def test_checkpoint_without_arch_loads_as_custom_cnn(tmp_path, dummy_dataset):
    from idefect.config import PreprocessingConfig
    from idefect.inference.predict import Predictor

    model = CustomCNN(num_classes=6, base_channels=8)
    torch.save({"state_dict": model.state_dict(), "class_names": [f"c{i}" for i in range(6)],
                "num_classes": 6, "image_size": 32,
                "cnn_config": {"in_channels": 1, "base_channels": 8, "dropout": 0.3},
                "preprocessing": PreprocessingConfig()}, tmp_path / "old.pt")
    p = Predictor(tmp_path / "old.pt")
    assert p.kind == "custom_cnn"
    assert p.predict(next(Path(dummy_dataset).rglob("*.png"))).label.startswith("c")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_bfloat16_autocast_trains_on_gpu():
    from idefect.training.train_cnn import _train_epoch

    model = build_cnn("resnet", num_classes=6, base_channels=8).cuda()
    data = torch.utils.data.TensorDataset(torch.randn(16, 1, 32, 32), torch.randint(0, 6, (16,)))
    loader = torch.utils.data.DataLoader(data, batch_size=8)
    opt = torch.optim.Adam(model.parameters())
    loss, acc = _train_epoch(model, loader, torch.nn.CrossEntropyLoss(), torch.device("cuda"), opt, amp=True)
    assert loss == loss and 0.0 <= acc <= 1.0  # finite loss, valid accuracy


def test_efficiency_benchmark_runs(tiny):
    from idefect.evaluation.efficiency import run_benchmark
    from idefect.training.train_cnn import train_cnn

    train_cnn(tiny)                                     # a tiny custom CNN
    results = Path(tiny.output.results_dir)
    rows = run_benchmark(tiny, results, results / "bench")
    r = next(r for r in rows if r["model"] == "custom_cnn")
    assert r["params_total"] >= r["params_trainable"] > 0
    assert r["cpu_latency_ms"] > 0 and r["cpu_throughput_ips"] > 0 and r["preprocess_ms"] > 0
    assert r["test_macro_f1"] is not None
    saved = json.loads((results / "bench" / "efficiency.json").read_text())
    assert saved["rows"][0]["model"] == "custom_cnn" and "environment" in saved


def test_gradcam_report_draws_every_cnn(tiny):
    from idefect.evaluation.cnn_report import make_cnn_figures
    from idefect.training.train_cnn import train_cnn

    for arch in ("custom_cnn", "resnet"):
        tiny.cnn.arch = arch
        train_cnn(tiny)
    out = make_cnn_figures(tiny, per_class=1, max_errors=2)
    assert {"gradcam_custom_cnn", "gradcam_resnet", "gradcam_comparison"} <= set(out)
    assert all(p.exists() and p.stat().st_size > 0 for p in out.values())


def test_subsample_is_stratified_seeded_and_from_train_only():
    from idefect.training.cnn_studies import subsample

    train = [(f"img{i}", i % 3) for i in range(60)]
    a, b = subsample(train, 0.25, 0), subsample(train, 0.25, 0)
    assert a == b and len(a) == 15
    assert sorted(lab for _, lab in a) == [0] * 5 + [1] * 5 + [2] * 5
    assert subsample(train, 0.25, 1) != a
    assert set(a) <= set(train)


def test_studies_use_validation_only(tiny, monkeypatch):
    import idefect.training.train_cnn as C
    from idefect.training.cnn_studies import data_curve, sensitivity_study

    seen = []
    real = C._train_once

    def spy(*a, **kw):
        seen.append(kw.get("evaluate_test", True))
        return real(*a, **kw)

    monkeypatch.setattr(C, "_train_once", spy)
    out = Path(tiny.output.results_dir) / "studies"
    res = sensitivity_study(tiny, [("custom_cnn", "full")], [0], out, sizes=(32,))
    tiny.data.image_size = 32
    curve = data_curve(tiny, [("custom_cnn", "full")], [("lbp", "logreg")], [0], out, fractions=(0.5,))
    assert seen and not any(seen)                       # test split never evaluated
    assert {r["variant"] for r in res["rows"]} >= {"32 px", "no CLAHE"}
    assert {r["branch"] for r in curve["rows"]} == {"cnn", "classical"}
    assert not list(out.rglob("*.pt"))                  # study checkpoints are not kept
