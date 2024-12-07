import torch
from typing import List

def cosine_similarity_matrix(A: List[torch.Tensor], B: List[torch.Tensor]) -> torch.Tensor:
    similarity_matrix = torch.cosine_similarity(A[:, None, :], B[None, :, :], dim=-1)

    return similarity_matrix

if __name__ == "__main__":
    n = 3