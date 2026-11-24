"""ImageNet transfer models: what is frozen really stays frozen, learning
rates per group, grayscale input handling, and save/reload (no download: the
tests build the backbones with random weights)."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from idefect.models.registry import artifact_stem, build_cnn, display_name  # noqa: E402
from idefect.models.transfer import BACKBONES, GrayToImageNet  # noqa: E402

ARCHS = list(BACKBONES)


def _trainable(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize("arch", ARCHS)
def test_grayscale_input_and_output_shape(arch):
    model = build_cnn(arch, num_classes=6, finetune="full").eval()
    with torch.no_grad():
        assert model(torch.rand(2, 1, 64, 64)).shape == (2, 6)


def test_gray_adapter_matches_imagenet_normalization():
    x = torch.full((1, 1, 2, 2), 0.5)
    y = GrayToImageNet()(x)
    assert y.shape == (1, 3, 2, 2)
    assert torch.allclose(y[0, :, 0, 0], torch.tensor([(0.5 - 0.485) / 0.229, (0.5 - 0.456) / 0.224,
                                                       (0.5 - 0.406) / 0.225]))


@pytest.mark.parametrize("arch", ARCHS)
def test_frozen_partial_full_train_nested_subsets(arch):
    frozen = _trainable(build_cnn(arch, 6, finetune="frozen"))
    partial = _trainable(build_cnn(arch, 6, finetune="partial"))
    full = _trainable(build_cnn(arch, 6, finetune="full"))
    assert frozen and all(n.startswith("net.") for n in frozen)
    assert frozen < partial < full          # strictly more layers train at each depth
    model = build_cnn(arch, 6, finetune="full")
    assert full == {n for n, _ in model.named_parameters()}


@pytest.mark.parametrize("arch", ARCHS)
def test_frozen_batchnorm_statistics_do_not_move(arch):
    model = build_cnn(arch, 6, finetune="frozen").train()
    bns = [m for m in model.net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns and all(not m.training for m in bns)
    before = bns[0].running_mean.clone()
    model(torch.rand(4, 1, 64, 64)).sum().backward()
    assert torch.equal(bns[0].running_mean, before)
    assert model.head.training


@pytest.mark.parametrize("mode", ["partial", "full"])
def test_pretrained_layers_learn_slower_than_the_head(mode):
    model = build_cnn("resnet18_imagenet", 6, finetune=mode)
    groups = model.param_groups(1e-3, 0.1)
    assert groups[0]["lr"] == 1e-3 and groups[1]["lr"] == pytest.approx(1e-4)
    ids = [{id(p) for p in g["params"]} for g in groups]
    assert not ids[0] & ids[1]
    assert ids[0] | ids[1] == {id(p) for p in model.parameters() if p.requires_grad}


def test_state_dict_has_no_duplicated_weights():
    model = build_cnn("resnet18_imagenet", 6, finetune="partial")
    keys = list(model.state_dict())
    assert len(keys) == len(set(keys))
    assert not any("_unfrozen" in k for k in keys)


def test_names():
    assert artifact_stem("resnet18_imagenet", "frozen") == "resnet18_imagenet_frozen"
    assert artifact_stem("resnet", "frozen") == "resnet"
    assert display_name("densenet121_imagenet", "partial") == "DenseNet-121 (ImageNet, partial)"


def test_transfer_model_trains_and_reloads(dummy_dataset, tmp_path):
    from idefect.config import Config
    from idefect.inference.predict import Predictor
    from idefect.training.train_cnn import train_cnn

    c = Config()
    c.data.root, c.data.manifest, c.data.image_size = str(dummy_dataset), "", 32
    c.data.train_frac, c.data.val_frac, c.data.test_frac = 0.5, 0.25, 0.25
    c.cnn.arch, c.cnn.finetune, c.cnn.pretrained = "mobilenet_v2_imagenet", "partial", False
    c.cnn.epochs, c.cnn.batch_size, c.cnn.num_workers, c.cnn.device, c.cnn.seeds = 1, 8, 0, "cpu", [0]
    c.data.duplicates.cache_dir, c.data.duplicates.review_file = str(tmp_path / "cache"), ""
    c.data.duplicates.top_k = 3
    c.output.results_dir = str(tmp_path / "results")
    c.output.figures_dir = str(tmp_path / "figures")
    c.output.logs_dir = str(tmp_path / "logs")
    m = train_cnn(c)
    stem = "mobilenet_v2_imagenet_partial"
    assert m["model"] == stem and m["finetune"] == "partial"
    assert m["n_params"] < m["n_params_total"]
    art = Path(c.output.results_dir) / f"{stem}.pt"
    assert (Path(c.output.results_dir) / "cnn_runs" / stem / "seed0" / f"{stem}.pt").exists()
    p = Predictor(art)
    assert p.kind == stem
    r = p.predict(next(Path(dummy_dataset).rglob("*.png")))
    assert abs(sum(r.probabilities.values()) - 1) < 1e-5


@pytest.mark.parametrize("arch", ["custom_cnn", "vgg", "resnet", "densenet", "mobilenet",
                                  "resnet18_imagenet", "densenet121_imagenet", "mobilenet_v2_imagenet"])
def test_gradcam_for_every_architecture(arch):
    import numpy as np

    from idefect.evaluation.gradcam import GradCAM

    model = build_cnn(arch, 6, finetune="frozen")
    model.train()
    cams, probs = GradCAM(model)(torch.rand(2, 1, 64, 64))
    assert cams.shape == (2, 64, 64) and probs.shape == (2, 6)
    assert np.isfinite(cams).all() and cams.min() >= 0 and cams.max() <= 1 + 1e-6
    assert model.training                    # the caller's mode is restored


def test_gradcam_points_at_the_evidence():
    """A model whose class-1 score is the brightness of the left half must
    produce a CAM concentrated on the left half."""
    from idefect.evaluation.gradcam import GradCAM, cam_energy_inside

    class LeftHalf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, 1, bias=False)
            torch.nn.init.ones_(self.conv.weight)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            a = torch.relu(self.conv(x))
            left = a.clone()
            left[..., left.shape[-1] // 2:] = 0
            return torch.cat([self.pool(a - left).flatten(1), self.pool(left).flatten(1) * 4], 1)

    x = torch.rand(1, 1, 32, 32) * 0.5 + 0.25
    cams, _ = GradCAM(LeftHalf())(x, class_idx=[1])
    mask = __import__("numpy").zeros((32, 32))
    mask[:, :16] = 1
    assert cam_energy_inside(cams[0], mask) > 0.95
