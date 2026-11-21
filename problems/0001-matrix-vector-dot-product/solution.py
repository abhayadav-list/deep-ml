import torch


def matrix_dot_vector(a, b) -> torch.Tensor:
    """
    Compute the product of matrix `a` and vector `b` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of length m, or tensor(-1) if dimensions mismatch.
    """
    a_t = torch.as_tensor(a, dtype=torch.float32)
    b_t = torch.as_tensor(b, dtype=torch.float32)

    # Ensure `a` is a non-empty 2D matrix and `b` is a 1D vector matching column count
    if a_t.ndim != 2 or b_t.ndim != 1 or a_t.shape[0] == 0 or a_t.shape[1] != b_t.shape[0]:
        return torch.tensor(-1)

    return torch.mv(a_t, b_t)