import pandas as pd

# Change this path to your extracted ml-100k folder
DATA_PATH = "ml-100k/"

# Load ratings
ratings = pd.read_csv(
    DATA_PATH + "u.data",
    sep="\t",
    names=["user", "item", "rating", "timestamp"]
)

# Load movie titles
movies = pd.read_csv(
    DATA_PATH + "u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["item", "title"]
)

# Merge ratings with movie titles
merged = pd.merge(ratings, movies, on="item")

# Keep only necessary columns
final_df = merged[["user", "item", "title", "rating"]]

# Save to your project data folder
final_df.to_csv("data/ratings_with_titles.csv", index=False)

print("CSV created successfully: data/ratings_with_titles.csv")