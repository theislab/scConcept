from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from concept.api import scConcept


def _loaded_concept():
    concept = scConcept()
    concept.cfg = OmegaConf.create({"datamodule": {"species": ["hsapiens"]}})
    concept.gene_mappings_path = "."
    concept.device = torch.device("cpu")
    concept.model = SimpleNamespace(logit_scale=torch.tensor(0.0))
    return concept


def test_estimate_mutual_information_matches_contrastive_formula():
    embs_1 = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]],
        dtype=torch.float32,
    )
    embs_2 = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)]],
        dtype=torch.float32,
    )
    concept = _loaded_concept()

    result = concept.estimate_mutual_information(embs_1, embs_2)

    logits = torch.mm(embs_1, embs_2.t())
    expected_loss = F.cross_entropy(logits, torch.arange(3)).item()
    expected_mutual_info = np.log(3) - expected_loss

    assert result == pytest.approx(expected_mutual_info, abs=1e-6)