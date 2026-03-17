import numpy as np
from lamin_dataloader import TokenizedDataset
from lamin_dataloader import GeneIdTokenizer


class MultiSpeciesTokenizer:
    """A single tokenizer that holds per-species gene → token-ID mappings.

    Each species has its own local token space (IDs ``0 .. vocab_size_i - 1``).
    Special tokens (CLS, PAD) are assumed to exist in every species' mapping
    under the same vocabulary keys and at the same integer positions.

    Args:
        species_gene_mappings: mapping ``{species_name: {gene_name: token_id}}``.
        CLS_VOCAB: vocabulary key for the CLS special token.
        PAD_VOCAB: vocabulary key for the PAD special token.
    """

    def __init__(
        self,
        species_gene_mappings: dict[str, dict],
        CLS_VOCAB: str = "<cls>",
        PAD_VOCAB: str = "<pad>",
    ):
        self.CLS_VOCAB = CLS_VOCAB
        self.PAD_VOCAB = PAD_VOCAB
        self._tokenizers: dict[str, GeneIdTokenizer] = {
            species: GeneIdTokenizer(gene_mapping, CLS_VOCAB, PAD_VOCAB)
            for species, gene_mapping in species_gene_mappings.items()
        }
        # Special-token IDs — caller must ensure they are consistent across species.
        _first = next(iter(self._tokenizers.values()))
        self.CLS_TOKEN = _first.CLS_TOKEN
        self.PAD_TOKEN = _first.PAD_TOKEN
        self.NOT_FOUND = _first.NOT_FOUND

    @property
    def species(self) -> list[str]:
        """Ordered list of species names registered with this tokenizer."""
        return list(self._tokenizers)

    @property
    def vocab_sizes(self) -> dict[str, int]:
        """Per-species vocabulary sizes, e.g. ``{"hsapiens": 20001, ...}``."""
        return {sp: len(tok.gene_mapping) for sp, tok in self._tokenizers.items()}

    def get_tokenizer(self, species: str) -> GeneIdTokenizer:
        """Return the underlying :class:`GeneIdTokenizer` for *species*."""
        return self._tokenizers[species]

    def encode(self, items, species: str) -> np.ndarray:
        """Encode gene names to token IDs in *species*' local token space."""
        return self._tokenizers[species].encode(items)

    def decode(self, items, species: str) -> np.ndarray:
        """Decode token IDs back to gene names for *species*."""
        return self._tokenizers[species].decode(items)


class MultiSpeciesTokenizedDataset(TokenizedDataset):
    """A :class:`TokenizedDataset` that is aware of per-dataset metadata.

    Args:
        metadata: A mapping from metadata key to a list of per-dataset values,
            e.g. ``{"species": ["hsapiens", "mmusculus"], "tissue": ["blood", "brain"]}``.
            The ``"species"`` key is required and is used to route tokenization to
            the correct species-specific vocabulary.
    """

    def __init__(self, metadata: dict[str, list], *args, **kwargs):
        self.metadata = metadata
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        output = super().__getitem__(idx)
        dataset_idx = self.collection.storage_idx[idx]
        for key, values in self.metadata.items():
            output[key] = values[dataset_idx]
        return output

    def _cache_tokenized_vars(self):
        species_list = self.metadata["species"]
        self.tokenized_vars = []
        for i, var_name in enumerate(self.collection.output_var_list):
            tokenized_var = self.tokenizer.encode(var_name, species_list[i])
            self.tokenized_vars.append(tokenized_var)

        self.masks = []
        self.tokenized_vars_masked = []
        for i, var_name in enumerate(self.collection.output_var_list):
            mask = self.tokenized_vars[i] != self.tokenizer.NOT_FOUND
            self.tokenized_vars_masked.append(self.tokenized_vars[i][mask])
            assert any(mask), f"dataset {self.collection.path_list[i]} has no token in common with vocabulary."
            self.masks.append(mask)