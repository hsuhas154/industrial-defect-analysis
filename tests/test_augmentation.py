"""Tests for CNN augmentation: label-safe ops and fresh randomness per epoch."""

import numpy as np
import pytest

from idefect.data.datasets import augment_image

torch = pytest.importorskip("torch")


def test_augment_image_ops():
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert np.array_equal(augment_image(img, False, False, 0), img)
    assert np.array_equal(augment_image(img, True, False, 0), np.fliplr(img))
    assert np.array_equal(augment_image(img, False, True, 0), np.flipud(img))
    assert np.array_equal(augment_image(img, False, False, 1), np.rot90(img))
    out = augment_image(img, True, True, 3)
    assert out.flags["C_CONTIGUOUS"] and sorted(out.ravel()) == sorted(img.ravel())


def test_augmentations_differ_across_epochs_with_workers(dummy_dataset):
    from torch.utils.data import DataLoader

    from idefect.config import PreprocessingConfig
    from idefect.data.datasets import NEUImageDataset, load_samples, seed_worker

    samples, _ = load_samples(dummy_dataset)
    ds = NEUImageDataset(samples[:8], PreprocessingConfig(clahe=False), 32,
                         augment=True, cache=True)
    gen = torch.Generator()
    gen.manual_seed(0)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                    worker_init_fn=seed_worker, generator=gen)
    epoch1 = next(iter(dl))[0]
    epoch2 = next(iter(dl))[0]
    assert not torch.equal(epoch1, epoch2), "augmentation repeated across epochs"
