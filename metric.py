"""
recall@k implementation from : https://github.com/leftthomas/CGD
"""

import torch
from typing import List


from typing import List, Tuple

def recall(query_features, query_labels, rank: List[int], gallery_features=None, gallery_labels=None) -> Tuple[List[float], torch.Tensor]:
    num_querys = len(query_labels)
    gallery_features = query_features if gallery_features is None else gallery_features

    cosine_matrix = query_features @ gallery_features.t()

    if gallery_labels is None:
        cosine_matrix.fill_diagonal_(-float('inf'))
        gallery_labels = query_labels

    # topkインデックスを取得
    topk_indices = cosine_matrix.topk(k=rank[-1], dim=-1, largest=True)[1]

    recall_list = []
    for r in rank:
        correct = (gallery_labels[topk_indices[:, 0:r]] == query_labels.unsqueeze(dim=-1)).any(dim=-1).float()
        recall_list.append((torch.sum(correct) / num_querys).item())

    # recall_list と topk_indices を返すように変更
    return recall_list, topk_indices

