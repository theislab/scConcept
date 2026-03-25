import numpy as np
import pytest

from concept.data.samplers import WithinGroupSampler


def test_within_group_sampler():
    """WithinGroupSampler must yield indices such that each batch has the same obs value."""
    # Three groups: 0..6 (obs=0), 7..13 (obs=1), 14..19 (obs=2)
    obs_list = np.array([0] * 7 + [1] * 7 + [2] * 6)
    batch_size = 3

    sampler = WithinGroupSampler(
        sampling_key=obs_list,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        stage="train",
    )

    indices = list(sampler)
    assert len(indices) > 0

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        if len(batch_indices) < batch_size:
            break
        batch_obs = obs_list[batch_indices]
        assert (batch_obs == batch_obs[0]).all(), (
            f"Batch at offset {i} has mixed obs values: {batch_obs}"
        )


def test_within_group_sampler_sample_groups_equally():
    """With sample_groups_equally=True, each group contributes the same number of samples."""
    # Unbalanced groups: 7 in group 0, 7 in group 1, 3 in group 2 (total 20)
    obs_list = np.array([0] * 7 + [1] * 7 + [2] * 3)
    batch_size = 3

    sampler = WithinGroupSampler(
        sampling_key=obs_list,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        sample_groups_equally=True,
        stage="train",
    )

    indices = list(sampler)
    assert len(indices) > 0

    # Each batch must still have uniform obs
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        if len(batch_indices) < batch_size:
            break
        batch_obs = obs_list[batch_indices]
        assert (batch_obs == batch_obs[0]).all(), (
            f"Batch at offset {i} has mixed obs values: {batch_obs}"
        )

    # Each group must contribute the same number of samples
    unique_obs = np.unique(obs_list)
    counts = [np.sum(obs_list[indices] == g) for g in unique_obs]
    assert len(set(counts)) == 1, (
        f"sample_groups_equally=True should balance groups; got counts {counts}"
    )
