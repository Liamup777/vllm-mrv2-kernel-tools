"""Small independent correctness checks; no NPU imports on the controller."""


def check_fill_num_accepted(args, kwargs, output):
    import torch

    mapping = kwargs["idx_mapping_ptr"].cpu().long()
    result = kwargs["num_accepted_ptr"].cpu()
    valid = mapping[mapping >= 0]
    assert (valid < result.numel()).all(), "mapping out of bounds"
    expected = torch.zeros_like(result)  # Example cases initialize output to zero.
    expected[valid] = kwargs["num_sampled"]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def check_postprocess_recoverssm_align(args, kwargs, output):
    """Check RecoverSSM state-column alignment and untouched output slots."""
    import torch

    del args, output

    def vector(name, dtype):
        value = kwargs[name]
        assert isinstance(value, torch.Tensor), f"{name} must be a tensor"
        assert value.ndim == 1, f"{name} must be one-dimensional"
        assert value.dtype == dtype, f"{name} must use {dtype}"
        assert value.is_contiguous(), f"{name} must be contiguous"
        return value

    idx_mapping = vector("idx_mapping_ptr", torch.int64)
    num_sampled = vector("num_sampled_ptr", torch.int32)
    num_computed = vector("num_computed_ptr", torch.int32)
    state_idx = vector("state_idx_ptr", torch.int32)
    num_accepted = vector("num_accepted_ptr", torch.int32)
    request_indices = kwargs.get("request_indices_ptr")
    if request_indices is None:
        batch_indices = range(idx_mapping.numel())
    else:
        request_indices = vector("request_indices_ptr", torch.int32)
        batch_indices = request_indices.detach().cpu().tolist()

    block_size = int(kwargs["MAMBA_BLOCK_SIZE"])
    table_width = int(kwargs["BLOCK_TABLE_WIDTH"])
    assert block_size > 0 and table_width > 0
    mapping = idx_mapping.detach().cpu().tolist()
    sampled = num_sampled.detach().cpu().tolist()
    computed = num_computed.detach().cpu().tolist()
    expected_state = torch.full_like(state_idx, -777, device="cpu")
    expected_accepted = torch.full_like(num_accepted, 9, device="cpu")
    written = set()

    for batch_idx in map(int, batch_indices):
        assert 0 <= batch_idx < len(mapping), "batch index out of bounds"
        assert batch_idx < len(sampled) and batch_idx < len(computed), "associated input too short"
        state_slot = int(mapping[batch_idx])
        if state_slot < 0:
            continue
        assert state_slot < state_idx.numel() and state_slot < num_accepted.numel(), "state index out of bounds"
        assert state_slot not in written, "duplicate state index creates a write race"
        written.add(state_slot)
        expected_state[state_slot] = min(
            (int(computed[batch_idx]) + int(sampled[batch_idx])) // block_size,
            table_width - 1,
        )
        expected_accepted[state_slot] = 1

    torch.testing.assert_close(state_idx.detach().cpu(), expected_state, rtol=0, atol=0)
    torch.testing.assert_close(num_accepted.detach().cpu(), expected_accepted, rtol=0, atol=0)
