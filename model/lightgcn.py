import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    """
    LightGCN: Simplified Graph Convolutional Network for Recommendation.
    Paper: https://arxiv.org/abs/2002.02126

    Improvements over original:
    - Multi-layer propagation (n_layers)
    - Layer combination via mean pooling of all layer outputs
    - Proper BPR loss support via forward pass
    - Dropout regularization on embeddings
    """

    def __init__(self, num_users, num_items, emb_dim=64, n_layers=3, dropout=0.1):
        super(LightGCN, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers
        self.dropout = dropout

        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def propagate(self, edge_index, all_emb):
        """One round of graph message passing with normalization."""
        row, col = edge_index
        agg = torch.zeros_like(all_emb)
        agg.index_add_(0, row, all_emb[col])
        return F.normalize(agg, p=2, dim=1)

    def forward(self, edge_index):
        """
        Multi-layer LightGCN propagation.
        Returns final embeddings as mean of all layer outputs (including layer 0).
        """
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight])

        # Apply dropout during training
        if self.training and self.dropout > 0:
            all_emb = F.dropout(all_emb, p=self.dropout)

        layer_outputs = [all_emb]

        for _ in range(self.n_layers):
            all_emb = self.propagate(edge_index, all_emb)
            layer_outputs.append(all_emb)

        # Mean pooling across layers (LightGCN paper Eq. 11)
        final_emb = torch.stack(layer_outputs, dim=0).mean(dim=0)

        return final_emb

    def get_user_item_emb(self, edge_index):
        """Returns (user_embeddings, item_embeddings) split after propagation."""
        final_emb = self.forward(edge_index)
        user_emb = final_emb[:self.num_users]
        item_emb = final_emb[self.num_users:]
        return user_emb, item_emb

    def bpr_loss(self, edge_index, users, pos_items, neg_items, lambda_reg=1e-4):
        """
        Bayesian Personalized Ranking loss.
        Encourages pos_item score > neg_item score for each user.
        """
        user_emb, item_emb = self.get_user_item_emb(edge_index)

        u = user_emb[users]
        pos = item_emb[pos_items]
        neg = item_emb[neg_items]

        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)

        bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

        # L2 regularization on embeddings (from initial, not propagated)
        reg = lambda_reg * (
            self.user_emb.weight[users].norm(2).pow(2) +
            self.item_emb.weight[pos_items].norm(2).pow(2) +
            self.item_emb.weight[neg_items].norm(2).pow(2)
        ) / len(users)

        return bpr + reg
