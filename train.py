import torch
from model.lightgcn import LightGCN

from torch_geometric.data import Data
import torch.serialization

# Allow torch_geometric Data loading (PyTorch 2.6+ fix)
torch.serialization.add_safe_globals([Data])


# =========================
# Load Graph Package
# =========================

pkg = torch.load("graph/graph_data.pt", weights_only=False)

graph = pkg["graph"]
num_users = pkg["num_users"]
num_items = pkg["num_items"]

edge_index = graph.edge_index


# =========================
# Create Model
# =========================

model = LightGCN(num_users, num_items)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50


# =========================
# Training Loop
# =========================

for epoch in range(EPOCHS):

    optimizer.zero_grad()

    embeddings = model(edge_index)

    # L2 Regularization Loss (simple baseline)
    loss = torch.mean(embeddings ** 2)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")


# =========================
# Save Model
# =========================

torch.save(model.state_dict(), "model/lightgcn_model.pt")

print("\nTraining Finished!")
print("Model saved: model/lightgcn_model.pt")