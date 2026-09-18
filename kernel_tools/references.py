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
