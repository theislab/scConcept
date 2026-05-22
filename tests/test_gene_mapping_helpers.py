import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from concept import scConcept


def mark_api_as_loaded(api, gene_mappings_path, species):
    api.model = object()
    api.gene_mappings_path = gene_mappings_path
    api.cfg = OmegaConf.create({"datamodule": {"species": species}})


def test_api_gene_name_to_id_helpers(tmp_path):
    pd.DataFrame(
        {
            "gene_id": ["ENSG000001", "ENSG000002"],
            "gene_name": ["TP53", "MALAT1"],
            "token": [2, 3],
        }
    ).to_csv(tmp_path / "hsapiens.csv", index=False)
    pd.DataFrame(
        {
            "gene_id": ["ENSMUSG000001"],
            "gene_name": ["Trp53"],
            "token": [2],
        }
    ).to_csv(tmp_path / "mmusculus.csv", index=False)

    api = scConcept()
    mark_api_as_loaded(api, tmp_path, ["hsapiens", "mmusculus"])

    assert api.species == ["hsapiens", "mmusculus"]
    assert api.get_gene_name_to_id_mapping("hsapiens") == {
        "TP53": "ENSG000001",
        "MALAT1": "ENSG000002",
    }

    mapped_genes = api.map_gene_names_to_ids("hsapiens", ["malat1", "UNKNOWN", "tp53"])

    assert mapped_genes[0] == "ENSG000002"
    assert np.isnan(mapped_genes[1])
    assert mapped_genes[2] == "ENSG000001"


def test_gene_name_to_id_helper_rejects_missing_required_columns(tmp_path):
    pd.DataFrame({"gene_id": ["ENSG000001"], "token": [2]}).to_csv(
        tmp_path / "hsapiens.csv",
        index=False,
    )

    api = scConcept()
    mark_api_as_loaded(api, tmp_path, ["hsapiens"])

    with pytest.raises(ValueError, match="missing required columns"):
        api.get_gene_name_to_id_mapping("hsapiens")


def test_gene_name_to_id_helper_keeps_first_duplicate_gene_name(tmp_path):
    pd.DataFrame(
        {
            "gene_id": ["ENSG000001", "ENSG000002", "ENSG000003"],
            "gene_name": ["TP53", "tp53", "MALAT1"],
            "token": [2, 3, 4],
        }
    ).to_csv(tmp_path / "hsapiens.csv", index=False)

    api = scConcept()
    mark_api_as_loaded(api, tmp_path, ["hsapiens"])

    assert api.get_gene_name_to_id_mapping("hsapiens") == {
        "TP53": "ENSG000001",
        "MALAT1": "ENSG000003",
    }


def test_gene_name_to_id_helpers_require_loaded_model(tmp_path):
    api = scConcept()

    with pytest.raises(ValueError, match="Call load_config_and_model"):
        api.species
