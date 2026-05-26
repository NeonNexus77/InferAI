"""
Axiomatic Error Clustering via Dimension Reduction.
Projects high-dimensional Sentence-BERT states onto 2D manifolds to map structural 
failures and zones of logic confusion for the research manuscript.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from classification.embedder import generate_embeddings

def run_error_clustering(dataset_path: str, output_image_path: str = "models/evaluation/regions_of_confusion.png"):
    print("[*] Loading Model components and verification matrices...")
    model = joblib.load(_ROOT / "models" / "nyaya_model.pkl")
    label_encoder = joblib.load(_ROOT / "models" / "label_encoder.pkl")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset requires 'text' and 'label' (ground truth) columns.")
        
    print(f"[*] Extracting high-dimensional embeddings for {len(df)} argument vectors...")
    embeddings = generate_embeddings(df["text"].tolist())
    
    print("[*] Running ML inference across structural test layers...")
    predictions = model.predict(embeddings)
    predicted_labels = label_encoder.inverse_transform(predictions)
    
    df["predicted"] = predicted_labels
    df["is_correct"] = df["label"] == df["predicted"]
    
    accuracy = df["is_correct"].mean() * 100
    print(f"[+] Current Pipeline Pipeline Accuracy: {accuracy:.2f}%")
    
    print("[*] Computing t-SNE structural manifold transformation (384-D -> 2D)...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(df) - 1), random_state=42, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    df["tsne_x"] = embeddings_2d[:, 0]
    df["tsne_y"] = embeddings_2d[:, 1]
    
    # Constructing the Plot
    plt.figure(figsize=(12, 10), dpi=300)
    
    # Plot correct predictions with soft transparency
    correct_mask = df["is_correct"] == True
    plt.scatter(
        df.loc[correct_mask, "tsne_x"],
        df.loc[correct_mask, "tsne_y"],
        c="gray",
        alpha=0.4,
        s=35,
        label="Correctly Classified Ground Truth"
    )
    
    # Highlight specific error states per Pramana class
    incorrect_df = df[~correct_mask]
    classes = unique_labels = df["label"].unique()
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    
    for i, label_name in enumerate(classes):
        class_errors = incorrect_df[incorrect_df["label"] == label_name]
        if not class_errors.empty:
            plt.scatter(
                class_errors["tsne_x"],
                class_errors["tsne_y"],
                edgecolors='black',
                linewidths=1.2,
                s=80,
                label=f"Misclassified Failure: {label_name}"
            )
            
            # Annotate specific extreme logical breakdown vectors
            for idx, row in class_errors.head(2).iterrows():
                plt.annotate(
                    f"True: {row['label']}\nPred: {row['predicted']}",
                    (row["tsne_x"], row["tsne_y"]),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7, lw=0.5)
                )

    plt.title("Axiomatic Error Clustering & Regions of Confusion (NyayaXAI)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=11)
    plt.ylabel("t-SNE Dimension 2", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", frameon=True, shadow=True)
    
    # Save mathematical visual asset
    output_path = _ROOT / output_image_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Axiomatic Visual Asset compiled successfully and saved to: {output_path}")

if __name__ == "__main__":
    # Can change to master_dataset.csv or real_world_dataset.csv depending on configuration
    target_dataset = str(_ROOT / "dataset" / "real_world_dataset.csv")
    run_error_clustering(target_dataset)