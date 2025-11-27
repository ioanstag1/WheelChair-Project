import os, glob, warnings, re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore")

# ----------------------------
# 1️⃣ Load all summary files
# ----------------------------
# csv_path = 'ADL1_2_3_features_poland_all.csv'
# out_dir = 'Results/Poland/with_knee/nested_LOO'
# csv_path = 'Features_All_noknee.csv'
# out_dir = 'Results/Brazil_Poland/no_knee/'

csv_path = 'Combined_Features_CLEANED.csv'
out_dir = 'Results/Brazil_Poland/Combined_Features_CLEANED/'

all_files = glob.glob(os.path.join(out_dir, "summary_*.csv"))
print(f"📂 Found {len(all_files)} summary files.")

dfs = []
for f in all_files:
    if os.path.getsize(f) == 0:
        print("⚠️ EMPTY FILE:", f)
        continue

    df = pd.read_csv(f)


    fname = os.path.basename(f).replace("summary_", "").replace(".csv", "")
 
    # Extract strategy (e.g., "svm_w") and k (e.g., "6")
    match = re.match(r"([a-zA-Z_]+)_k(\d+)", fname)
    if match:
        strategy, k = match.groups()
        df["strategy"] = strategy
        df["k"] = int(k)
    else:
        df["strategy"] = fname
        df["k"] = None

    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)
print(f"✅ Combined {len(all_df)} rows from {len(all_files)} files.")

# ----------------------------
# 2️⃣ Find Top 10 Models per Strategy
# ----------------------------
# ----------------------------
# 2️⃣ Find Top 10 Models per Strategy
# ----------------------------
top10_per_strategy = (
    all_df.sort_values(by=["strategy", "bal_acc_mean", "auc"], ascending=[True, False, False])  # Sort by bal_acc_mean and auc
    .groupby("strategy")
    .head(10)
)

print("\n🔥 Top 10 Models per Feature Selection Strategy:")
for strat, sub in top10_per_strategy.groupby("strategy"):
    print(f"\n=== {strat.upper()} ===")
    print(sub[["model", "k", "bal_acc_mean", "acc_mean", "f1_mean", "auc"]].round(3))


# ----------------------------
# 3️⃣ Find the best model overall
# ----------------------------
# Sort by 'bal_acc_mean' first (descending), then 'auc' second (descending)
best_overall = all_df.sort_values(by=["bal_acc_mean", "auc"], ascending=[False, False]).iloc[0]

# Show the best model
cols_to_show = ["model", "strategy", "k", "bal_acc_mean", "auc", "f1_mean"]
best_row = best_overall[cols_to_show].copy()
for c in ["bal_acc_mean", "auc", "f1_mean"]:
    best_row[c] = round(float(best_row[c]), 3)

print("\n🏆 Best overall model (prioritizing AUC if accuracies are tied):")
print(best_row)

# ----------------------------
# 6️⃣ Extract and visualize feature frequencies for the true best model per model type
# ----------------------------
import ast
import matplotlib
matplotlib.use("Agg")  # no GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np

# --- Load feature names ---
# csv_path = 'Features_All.csv'
# out_dir = 'Results/Brazil_Poland/with_knee/nested_LOO_3class'
# csv_path = 'Features_All_noknee.csv'
# out_dir = 'Results/Brazil_Poland/no_knee/'
df = pd.read_csv(csv_path)
#df = df.drop(columns=["HeadRoll_RMS_ADL2"])
df["binary_label"] = df["label"].replace({2: 1})
X = df.select_dtypes(include=[np.number]).drop(columns=["label", "binary_label"], errors="ignore")
feature_names = list(X.columns)
#print(feature_names)
print(f"✅ Loaded {len(feature_names)} feature names from {csv_path}")


# --- Helper functions ---
def plot_feature_hist(df_plot, model, strategy, k, out_dir):
    """Plot and save histogram of all features for a given model"""
    plt.figure(figsize=(10, max(6, len(df_plot) * 0.2)))
    sns.barplot(y="feature_name", x=model, data=df_plot, palette="viridis")
    plt.title(f"Feature Selection Frequency for {model}\n({strategy}, k={k})")
    plt.xlabel("Selection Count")
    plt.ylabel("Feature Name")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(out_dir, f"ALL_features_hist_{model}_{strategy}_k{k}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"📊 Saved histogram: {os.path.basename(fig_path)}")


def extract_all_features_for_model(model, strategy, k, out_dir):
    """Extract all feature frequencies for given model"""
    freq_path = os.path.join(out_dir, f"feature_frequency_{strategy}_k{k}_ALL.csv")
    if not os.path.exists(freq_path):
        print(f"⚠️ No frequency file found for {model} ({strategy}_k{k})")
        return None

    df_freq = pd.read_csv(freq_path)
    if model not in df_freq.columns:
        print(f"⚠️ Model {model} not found in {freq_path}")
        return None

    # Keep relevant columns
    df_freq = df_freq[["feature_index", model]].dropna()
    df_freq = df_freq.sort_values(by=model, ascending=False).reset_index(drop=True)

    # Map index to name
    df_freq["feature_name"] = df_freq["feature_index"].apply(
        lambda i: feature_names[int(i) - 1] if 1 <= int(i) <= len(feature_names) else f"unknown_{i}"
    )

    # Save CSV
    csv_path = os.path.join(out_dir, f"ALL_features_{model}_{strategy}_k{k}.csv")
    df_freq.to_csv(csv_path, index=False)
    print(f"✅ Saved all feature frequencies: {os.path.basename(csv_path)}")

    # Plot histogram
    plot_feature_hist(df_freq, model, strategy, k, out_dir)

    return df_freq

# --- 1️⃣ Find true best model per model type ---
print("\n🔍 Finding true best configuration per model type...")

# Sort by 'bal_acc_mean' (descending), then 'auc' (descending) to break ties
best_per_model = (
    all_df.sort_values(["model", "bal_acc_mean", "auc"], ascending=[True, False, False])  # Sort by bal_acc_mean first, then auc
          .groupby("model")
          .first()  # Picks the top row per model
          .reset_index()
)

print("\n🏆 Best per model type:")
print(best_per_model[["model", "strategy", "k", "bal_acc_mean", "auc"]])

# --- 2️⃣ Extract and visualize features for each best model ---
all_results = []

for _, row in best_per_model.iterrows():
    model = row["model"]
    strategy = row["strategy"]
    k = int(row["k"])
    print(f"\n➡️ Analyzing {model} (strategy={strategy}, k={k})")

    df_features = extract_all_features_for_model(model, strategy, k, out_dir)
    if df_features is not None:
        df_features["model"] = model
        df_features["strategy"] = strategy
        df_features["k"] = k
        all_results.append(df_features)

# --- 3️⃣ Save combined summary ---
if all_results:
    feat_summary = pd.concat(all_results, ignore_index=True)
    out_summary = os.path.join(out_dir, "ALL_features_per_model_summary.csv")
    feat_summary.to_csv(out_summary, index=False)
    print(f"\n✅ Saved combined summary: {out_summary}")

# --- 4️⃣ Global best model ---
print("\n🌟 Global best model:")
print(best_row)



# ----------------------------
# 🧩 Best results of each model in each strategy
# ----------------------------

# Group by both strategy and model → pick the best configuration (max bal_acc_mean)
# Group by both strategy and model → pick the best configuration (max bal_acc_mean and use auc to break ties)
best_model_strategy = (
    all_df.sort_values(by=["bal_acc_mean", "auc"], ascending=[False, False])  # Sort first by bal_acc_mean, then auc
          .groupby(["strategy", "model"])
          .first()  # picks top row per (strategy, model)
          .reset_index()
)

# Display the results
print("\n🏆 Best result of each model in each strategy:")
print(best_model_strategy[["strategy", "model", "k", "bal_acc_mean", "auc", "f1_mean"]])

# Save to CSV for full inspection
best_path = os.path.join(out_dir, "best_results_per_model_per_strategy.csv")
best_model_strategy.to_csv(best_path, index=False)
print(f"✅ Saved: {best_path}")

# Create a pivot table for both bal_acc_mean and auc
pivot = best_model_strategy.pivot(index="model", columns="strategy", values=["bal_acc_mean", "auc"])

# Plotting heatmap for both bal_acc_mean and auc
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Heatmap for bal_acc_mean
sns.heatmap(pivot["bal_acc_mean"], annot=True, fmt=".3f", cmap="YlGnBu", ax=ax1)
ax1.set_title("Best Balanced Accuracy of Each Model per Strategy")
ax1.set_xlabel("Strategy")
ax1.set_ylabel("Model")

# Heatmap for auc
sns.heatmap(pivot["auc"], annot=True, fmt=".3f", cmap="YlGnBu", ax=ax2)
ax2.set_title("Best AUC of Each Model per Strategy")
ax2.set_xlabel("Strategy")
ax2.set_ylabel("Model")

plt.tight_layout()

# Save the heatmaps
heatmap_path = os.path.join(out_dir, "heatmap_best_model_per_strategy.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()
print(f"📊 Saved heatmap: {heatmap_path}")