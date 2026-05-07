from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from concept.api import scConcept


def _loaded_concept(monkeypatch, embeddings_by_panel):
    concept = scConcept()
    concept.cfg = OmegaConf.create({"datamodule": {"species": ["hsapiens"]}})
    concept.gene_mappings_path = Path(".")
    concept.device = torch.device("cpu")
    concept.model = SimpleNamespace(logit_scale=torch.tensor(0.0))

    def fake_extract_embeddings(adata, **kwargs):
        assert kwargs["return_type"] == "torch"
        key = tuple(adata.var_names.astype(str))
        return {"cls_cell_emb": embeddings_by_panel[key]}

    monkeypatch.setattr(concept, "extract_embeddings", fake_extract_embeddings)
    return concept


def _loaded_concept_from_obs(monkeypatch):
    concept = scConcept()
    concept.cfg = OmegaConf.create({"datamodule": {"species": ["hsapiens"]}})
    concept.gene_mappings_path = Path(".")
    concept.device = torch.device("cpu")
    concept.model = SimpleNamespace(logit_scale=torch.tensor(0.0))
    observed_obs_names = []

    def fake_extract_embeddings(adata, **kwargs):
        assert kwargs["return_type"] == "torch"
        observed_obs_names.append(tuple(adata.obs_names.astype(str)))
        return {"cls_cell_emb": torch.eye(adata.n_obs, dtype=torch.float32)}

    monkeypatch.setattr(concept, "extract_embeddings", fake_extract_embeddings)
    return concept, observed_obs_names


def test_estimate_mutual_information_matches_contrastive_formula(monkeypatch):
    adata = ad.AnnData(np.ones((3, 4)))
    adata.var_names = ["gene_a", "gene_b", "gene_c", "gene_d"]

    panel_1 = ["gene_a", "gene_c"]
    panel_2 = ["gene_b", "gene_d"]
    embs_1 = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]],
        dtype=torch.float32,
    )
    embs_2 = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)]],
        dtype=torch.float32,
    )
    concept = _loaded_concept(
        monkeypatch,
        {
            tuple(panel_1): embs_1,
            tuple(panel_2): embs_2,
        },
    )

    result = concept.estimate_mutual_information(
        adata,
        panel_1,
        panel_2,
        species="hsapiens",
        batch_size=2,
    )

    logits = torch.mm(embs_1, embs_2.t())
    expected_loss = F.cross_entropy(logits, torch.arange(3)).item()
    expected_mutual_info = np.log(3) - expected_loss

    assert result["mutual_info"] == pytest.approx(expected_mutual_info, abs=1e-6)
    assert result["contrastive_loss"] == pytest.approx(expected_loss, abs=1e-6)
    assert result["n_cells"] == 3
    assert result["panel_size_1"] == 2
    assert result["panel_size_2"] == 2
    assert result["features_size_1"] == 2
    assert result["features_size_2"] == 2
    np.testing.assert_array_equal(result["features_1"], np.asarray(panel_1))
    np.testing.assert_array_equal(result["features_2"], np.asarray(panel_2))


def test_estimate_mutual_information_can_match_panels_by_gene_id_column(monkeypatch):
    adata = ad.AnnData(np.ones((2, 3)))
    adata.var_names = ["symbol_a", "symbol_b", "symbol_c"]
    adata.var["ensembl_id"] = ["ENSG1", "ENSG2", "ENSG3"]

    concept = _loaded_concept(
        monkeypatch,
        {
            ("symbol_a", "symbol_c"): torch.eye(2, dtype=torch.float32),
            ("symbol_b",): torch.eye(2, dtype=torch.float32),
        },
    )

    result = concept.estimate_mutual_information(
        adata,
        ["ENSG3", "ENSG1", "ENSG4"],
        ["ENSG2"],
        species="hsapiens",
        gene_id_column="ensembl_id",
    )

    assert result["panel_size_1"] == 3
    assert result["features_size_1"] == 2
    assert result["features_size_2"] == 1
    np.testing.assert_array_equal(result["features_1"], np.asarray(["ENSG1", "ENSG3"]))


def test_estimate_mutual_information_all_genes_uses_gene_id_column(monkeypatch):
    adata = ad.AnnData(np.ones((2, 2)))
    adata.var_names = ["symbol_a", "symbol_b"]
    adata.var["ensembl_id"] = ["ENSG1", "ENSG2"]

    concept = _loaded_concept(
        monkeypatch,
        {
            ("symbol_a", "symbol_b"): torch.eye(2, dtype=torch.float32),
        },
    )

    result = concept.estimate_mutual_information(
        adata,
        "all_genes",
        "all_genes",
        species="hsapiens",
        gene_id_column="ensembl_id",
    )

    assert result["panel_size_1"] == 2
    np.testing.assert_array_equal(result["features_1"], np.asarray(["ENSG1", "ENSG2"]))


def test_estimate_mutual_information_samples_cells_before_embedding_extraction(monkeypatch):
    adata = ad.AnnData(np.ones((5, 2)))
    adata.obs_names = [f"cell_{i}" for i in range(5)]
    adata.var_names = ["gene_a", "gene_b"]
    concept, observed_obs_names = _loaded_concept_from_obs(monkeypatch)

    result = concept.estimate_mutual_information(
        adata,
        ["gene_a"],
        ["gene_b"],
        species="hsapiens",
        estimate_size=3,
        random_seed=7,
    )

    expected_indices = np.random.default_rng(seed=7).choice(adata.n_obs, 3, replace=False)
    expected_obs_names = tuple(adata.obs_names[expected_indices].astype(str))

    assert result["n_cells"] == 3
    assert observed_obs_names == [expected_obs_names, expected_obs_names]
