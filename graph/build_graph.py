import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:\\Users\\redde\\OneDrive\\Desktop\\GNN-Recommender\\data\\ratings_with_titles.csv")

# Encode user and item ids
user_enc = LabelEncoder()
item_enc = LabelEncoder()

df["user_id"] = user_enc.fit_transform(df["user"])
df["item_id"] = item_enc.fit_transform(df["item"])

num_users = df["user_id"].nunique()
num_items = df["item_id"].nunique()

users = df["user_id"].values
items = df["item_id"].values + num_users  # shift items

edges = []

for u, i in zip(users, items):
    edges.append([u, i])
    edges.append([i, u])

edge_index = torch.tensor(edges, dtype=torch.long).t()

data = Data(edge_index=edge_index)

# Save everything
torch.save({
    "graph": data,
    "num_users": num_users,
    "num_items": num_items
}, "graph/graph_data.pt")

print("Graph rebuilt successfully!")