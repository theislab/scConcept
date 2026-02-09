import numpy as np
import pytest

from concept.data.samplers import WithinGroupSampler


def test_within_group_sampler():
    """WithinGroupSampler must yield indices such that each batch has the same obs value."""
    # Three groups: 0..6 (obs=0), 7..13 (obs=1), 14..19 (obs=2)
    obs_list = np.array([0] * 7 + [1] * 7 + [2] * 6)
    batch_size = 3

    sampler = WithinGroupSampler(
        obs_list=obs_list,
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
