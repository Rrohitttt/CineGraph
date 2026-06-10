"""
train_improved.py — LightGCN training with BPR loss + negative sampling

Improvements over original train.py:
- Replaces L2 embedding loss with BPR (Bayesian Personalized Ranking) loss
- Adds mini-batch training with negative sampling
- Adds train/val split for tracking generalization
- Saves best model by validation loss
- Prints richer training progress
"""

import torch
import random
import numpy as np
from torch_geometric.data import Data
import torch.serialization
from model.lightgcn_improved import LightGCN

torch.serialization.add_safe_globals([Data])

# ===========================
# Config
# ===========================
EPOCHS = 100
LR = 0.001
EMB_DIM = 64
N_LAYERS = 3
DROPOUT = 0.1
BATCH_SIZE = 1024
LAMBDA_REG = 1e-4
VAL_SPLIT = 0.1
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ===========================
# Load Graph
# ===========================
pkg = torch.load("graph/graph_data.pt", weights_only=False)
graph = pkg["graph"]
num_users = pkg["num_users"]
num_items = pkg["num_items"]
edge_index = graph.edge_index

print(f"Loaded graph: {num_users} users, {num_items} items")
print(f"Total edges: {edge_index.shape[1]}")


# ===========================
# Build interaction set for negative sampling
# ===========================
# edge_index rows: [user_nodes, item_nodes (shifted by num_users)]
user_nodes = edge_index[0]
item_nodes = edge_index[1]

# Filter only user->item edges (user_node < num_users, item_node >= num_users)
mask = (user_nodes < num_users) & (item_nodes >= num_users)
u_ids = user_nodes[mask].tolist()
i_ids = (item_nodes[mask] - num_users).tolist()  # unshift

interactions = list(zip(u_ids, i_ids))
print(f"Interaction pairs: {len(interactions)}")

# Train/val split
random.shuffle(interactions)
val_size = int(len(interactions) * VAL_SPLIT)
val_pairs = interactions[:val_size]
train_pairs = interactions[val_size:]

# User item sets for fast negative sampling
user_item_set = set(interactions)
item_list = list(range(num_items))


def sample_negatives(users):
    negs = []
    for u in users:
        while True:
            neg = random.randint(0, num_items - 1)
            if (u, neg) not in user_item_set:
                negs.append(neg)
                break
    return negs


# ===========================
# Model + Optimizer
# ===========================
model = LightGCN(num_users, num_items, emb_dim=EMB_DIM, n_layers=N_LAYERS, dropout=DROPOUT)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

best_val_loss = float("inf")

# ===========================
# Training Loop
# ===========================
for epoch in range(1, EPOCHS + 1):
    model.train()
    random.shuffle(train_pairs)
    total_loss = 0
    num_batches = 0

    for start in range(0, len(train_pairs), BATCH_SIZE):
        batch = train_pairs[start:start + BATCH_SIZE]
        users_b = [p[0] for p in batch]
        pos_items_b = [p[1] for p in batch]
        neg_items_b = sample_negatives(users_b)

        users_t = torch.tensor(users_b, dtype=torch.long)
        pos_t = torch.tensor(pos_items_b, dtype=torch.long)
        neg_t = torch.tensor(neg_items_b, dtype=torch.long)

        optimizer.zero_grad()
        loss = model.bpr_loss(edge_index, users_t, pos_t, neg_t, lambda_reg=LAMBDA_REG)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    scheduler.step()
    avg_train_loss = total_loss / max(num_batches, 1)

    # Validation
    model.eval()
    with torch.no_grad():
        val_users = torch.tensor([p[0] for p in val_pairs[:512]], dtype=torch.long)
        val_pos = torch.tensor([p[1] for p in val_pairs[:512]], dtype=torch.long)
        val_neg = torch.tensor(sample_negatives([p[0] for p in val_pairs[:512]]), dtype=torch.long)
        val_loss = model.bpr_loss(edge_index, val_users, val_pos, val_neg, lambda_reg=LAMBDA_REG).item()

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "model/lightgcn_model_improved.pt")

print(f"\nTraining finished! Best val loss: {best_val_loss:.4f}")
print("Model saved: model/lightgcn_model_improved.pt")
