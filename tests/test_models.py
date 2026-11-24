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
