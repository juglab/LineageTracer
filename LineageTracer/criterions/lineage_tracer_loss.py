import numpy as np
import torch
import torch.nn as nn


class LineageTracerLoss(nn.Module):

  def __init__(self, margin=0.2):
    super().__init__()

    print('Created  loss function with: margin: {}'.format(margin))
    print('*************************')
    self.margin = margin
    self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)

  def forward(self, embeddings, targets):
    """
        Args:
            embeddings: feature matrix with shape ~72 x 32
            targets: ground truth labels with shape ~72 [1 1 1 2 2 2 3 3 3 4 4 4 ... 24 24 24]
        """

    n = embeddings.size(0)
    # just compute pairwise L_{2} distance between embeddings
    dist = torch.pow(embeddings, 2).sum(dim=1, keepdim=True).expand(n,
                                                                    n)  # 72 x 72 --> find norm of embeddings and repeat
    dist = dist + dist.t()
    dist.addmm_(1, -2, embeddings,
                embeddings.t())  # stands for add matrix multiplication in place -2*dist + embeddings*embeddings_tranposed
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    # For each anchor, find the hardest positive and negative
    mask = targets.expand(n, n).eq(targets.expand(n, n).t())
    loss = torch.zeros([1]).cuda()
    if mask.float().unique().shape[0] > 1:
      dist_anchor_positive, dist_anchor_negative = [], []
      for i in range(n):
        dist_anchor_positive.append(dist[i][mask[i]].max().unsqueeze(0))
        dist_anchor_negative.append(dist[i][mask[i] == 0].min().unsqueeze(0))
      dist_anchor_positive = torch.cat(dist_anchor_positive)
      dist_anchor_negative = torch.cat(dist_anchor_negative)
      # Compute ranking hinge loss
      y = torch.ones_like(dist_anchor_negative)
      loss = self.ranking_loss(dist_anchor_negative, dist_anchor_positive, y).unsqueeze(
        0)  # hinge loss is computed as loss(x1, x2, y) where x1, x2 and y are 1d tensors of size N!

    return loss
