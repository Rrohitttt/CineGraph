# 🎬 CineGraph — GNN Movie Recommendation System

A Graph Neural Network (LightGCN) based movie recommender built on MovieLens 100K.

## 🚀 Quick Start

### Step 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Install PyTorch Geometric (run this separately)
```bash
pip install torch_geometric
```

### Step 3 — Run the app
```bash
streamlit run app/app.py
```

Open browser at: **http://localhost:8501**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 Personalized Recs | GNN-based recommendations per user |
| 🔍 Movie Search | Search any movie → find similar titles instantly |
| 🎞️ Movie Posters | Styled poster cards for every movie |
| 🔥 Trending | Most popular + highest rated sections |
| 📼 User History | Watch history + genre breakdown chart |
| ⬇️ CSV Export | Download recs, search results, history |
| 🎭 Genre Filter | Filter recommendations by genre |
| 👥 Similar Users | Discover users with similar taste |

---

## 📁 Project Structure

```
softcomputing/
├── app/
│   └── app.py                  ← Main Streamlit app (all features)
├── data/
│   └── ratings_with_titles.csv ← MovieLens with titles
├── graph/
│   ├── build_graph.py          ← Build graph from ratings
│   └── graph_data.pt           ← Pre-built graph (ready to use)
├── ml-100k/                    ← Raw MovieLens 100K dataset
├── model/
│   ├── lightgcn.py             ← LightGCN model (3-layer)
│   ├── lightgcn_model.pt       ← Pre-trained weights
│   └── __init__.py
├── train.py                    ← Original training script
├── train_improved.py           ← BPR loss training (recommended)
├── requirements.txt
└── README.md
```

---

## 🔁 Optional: Retrain the Model (Better Accuracy)

```bash
python train_improved.py
```

Uses BPR loss + negative sampling. Saves best model automatically.

---

## ❓ Common Issues

**`ModuleNotFoundError: torch_geometric`**
```bash
pip install torch_geometric
```

**App won't start**
Make sure you're running from inside the `softcomputing/` folder.

**Posters look like colored blocks**
That's expected — they are styled placeholders. Add a TMDB API key in the sidebar for real posters (optional).
