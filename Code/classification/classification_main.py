# ============================================================
# 🧠 Nested Leave-One-Out CV with multiple feature selection strategies
# Multiclass version (3-class classification)
# Includes SVM weights, Leave-one-out NB importance, PCA, t-test, RFE, and Combined.
# Based on Zhao et al. (2017) methodology.
# ============================================================
from tqdm import tqdm

import os, warnings, time, glob
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score,
    precision_score, f1_score, roc_auc_score
)
from sklearn.base import BaseEstimator, TransformerMixin

# ============================================================
# 0️⃣ LOAD DATA
# ============================================================
# csv_path = 'Features_All.csv'
# out_dir = 'Results/Brazil_Poland/with_knee/nested_LOO_3class/'

csv_path = 'Combined_Features_CLEANED_kneerom.csv'
out_dir = 'Results/Brazil_Poland/Combined_Features_CLEANED_kneerom/'


os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)
#df = df.drop(columns=["HeadRoll_RMS_ADL2"])
y = df['label'].values  # keep original 3-class labels
X = df.select_dtypes(include=[np.number]).drop(columns=['label'], errors='ignore')

print(f"✅ Loaded {len(X)} samples and {X.shape[1]} numeric features.")
print("Class balance:", dict(zip(*np.unique(y, return_counts=True))))

# ============================================================
# 1️⃣ FEATURE SELECTION HELPERS
# ============================================================
def _svm_weight_topk(Xtr, ytr, k, C=1.0):
    Xs = StandardScaler().fit_transform(Xtr)
    clf = SVC(kernel='linear', C=C)
    clf.fit(Xs, ytr)
    # average abs weights across all classes for multiclass SVM
    w = np.mean(np.abs(clf.coef_), axis=0)
    return np.argsort(w)[::-1][:min(k, Xtr.shape[1])]

def _loo_drop_nb_topk(Xtr, ytr, k, cv_splits=3):
    from sklearn.model_selection import cross_val_score
    base = GaussianNB()
    inner = StratifiedKFold(n_splits=min(cv_splits, np.unique(ytr, return_counts=True)[1].min()),
                            shuffle=True, random_state=1)
    base_acc = np.mean(cross_val_score(base, Xtr, ytr, cv=inner, scoring='balanced_accuracy'))
    drops = []
    for i in range(Xtr.shape[1]):
        Xi = Xtr.drop(Xtr.columns[i], axis=1)
        acc_i = np.mean(cross_val_score(base, Xi, ytr, cv=inner, scoring='balanced_accuracy'))
        drops.append(base_acc - acc_i)
    return np.argsort(drops)[::-1][:min(k, Xtr.shape[1])]

def _pca_topk(Xtr, ytr, k):
    Xs = StandardScaler().fit_transform(Xtr)
    pca = PCA()
    pca.fit(Xs)
    load = np.abs(pca.components_[0])
    return np.argsort(load)[::-1][:min(k, Xtr.shape[1])]

def _ttest_topk(Xtr, ytr, k):
    sel = SelectKBest(score_func=f_classif, k=min(k, Xtr.shape[1]))
    sel.fit(Xtr, ytr)
    return np.where(sel.get_support())[0]

def _rfe_topk(Xtr, ytr, k):
    base = SVC(kernel='linear', C=1.0)
    Xs = StandardScaler().fit_transform(Xtr)
    sel = RFE(base, n_features_to_select=min(k, Xtr.shape[1]), step=1)
    sel.fit(Xs, ytr)
    return np.where(sel.support_)[0]

# ============================================================
# 2️⃣ CUSTOM FEATURE SELECTOR
# ============================================================
class StrategySelector(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='none', k=10):
        self.strategy = strategy
        self.k = k
        self.feature_indices_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if not hasattr(X, 'columns'):
            X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        k = int(self.k)

        if self.strategy == 'none':
            self.feature_indices_ = np.arange(X.shape[1])
        elif self.strategy == 'svm_w':
            self.feature_indices_ = _svm_weight_topk(X, y, k)
        elif self.strategy == 'loo_drop':
            self.feature_indices_ = _loo_drop_nb_topk(X, y, k)
        elif self.strategy == 'pca_w':
            self.feature_indices_ = _pca_topk(X, y, k)
        elif self.strategy == 'ttest':
            self.feature_indices_ = _ttest_topk(X, y, k)
        elif self.strategy == 'rfe':
            self.feature_indices_ = _rfe_topk(X, y, k)
        elif self.strategy == 'combo':
            idx_svm = _svm_weight_topk(X, y, k=max(8, k))
            idx_loo = _loo_drop_nb_topk(X, y, k=max(10, k))
            idx_pca = _pca_topk(X, y, k=max(3, k))
            combined = list(np.unique(np.concatenate([
                idx_svm[:8],
                np.intersect1d(idx_pca[:3], idx_loo[:10])
            ])))
            if len(combined) < k:
                extra = [i for i in idx_svm if i not in combined][: k - len(combined)]
                combined += extra
            self.feature_indices_ = np.array(combined, dtype=int)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        return self

    def transform(self, X):
        if not hasattr(X, 'columns'):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X.iloc[:, self.feature_indices_].values

# ============================================================
# 3️⃣ MODELS + PARAMETER GRIDS
# ============================================================
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

random_state = 42

model_spaces = {
    # "NB": {
    #     "estimator": GaussianNB(),  # NB doesn't support class_weight directly
    #     "params": {}
    # },
    "SVM_linear": {
        "estimator": SVC(kernel='linear', probability=True, class_weight='balanced', random_state=random_state),
        "params": {"clf__C": [0.1, 1.0, 10.0]}
    },
    "SVM_rbf": {
        "estimator": SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=random_state),
        "params": {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.1]}
    },
    "LogReg_L1": {
        "estimator": LogisticRegression(
            max_iter=2000,
            solver='liblinear',
            penalty='l1',
            class_weight='balanced',
            random_state=random_state
        ),
        "params": {"clf__C": [0.1, 1.0, 10.0]}
    },
    "RF": {
        "estimator": RandomForestClassifier(class_weight='balanced', random_state=random_state),
        "params": {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 4, 6]}
    }
}


# ============================================================
# 4️⃣ EXPERIMENT CONFIG
# ============================================================
strategies = ['ttest', 'rfe']
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
outer = LeaveOneOut()

# ============================================================
# 5️⃣ RUN NESTED LOO FOR ALL STRATEGIES AND FEATURE COUNTS
# ============================================================

for strategy in strategies:
    print(f"\n============================")
    print(f"🚀 Running strategy: {strategy}")
    print(f"============================")

    k_values = [X.shape[1]] if strategy == "none" else range(5, X.shape[1])

    for k_val in k_values:
        print(f"\n🎯 Evaluating k={k_val} features")

        rows = []
        preds_per_model = {m: {"y_true": [], "y_prob": [], "y_pred": []} for m in model_spaces.keys()}
        feat_freqs = {m: Counter() for m in model_spaces.keys()}

        # --- OUTER LOO WITH PROGRESS BAR ---
        pbar = tqdm(total=len(X), desc=f"{strategy} | k={k_val}", ncols=110)

        for i, (tr_idx, te_idx) in enumerate(outer.split(X, y), 1):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            for model_name, spec in model_spaces.items():

                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("select", StrategySelector(strategy=strategy, k=k_val)),
                    ("clf", spec["estimator"])
                ])

                # param grid
                grid = {"select__k": [k_val]}
                for p, vals in spec["params"].items():
                    grid[p] = vals

                gs = GridSearchCV(
                    pipe, [grid],
                    scoring="balanced_accuracy",
                    cv=inner,
                    n_jobs=-1,
                    refit=True
                )
                gs.fit(X_tr, y_tr)
                best = gs.best_estimator_

                y_pred = best.predict(X_te)

                try:
                    y_prob = best.predict_proba(X_te)[0]
                except:
                    y_prob = np.ones(len(np.unique(y))) / len(np.unique(y))

                # record selected features
                feat_idx = best.named_steps["select"].feature_indices_
                feat_freqs[model_name].update(feat_idx)

                preds_per_model[model_name]["y_true"].append(int(y_te[0]))
                preds_per_model[model_name]["y_pred"].append(int(y_pred[0]))
                preds_per_model[model_name]["y_prob"].append(y_prob)

                rows.append({
                    "fold": i,
                    "model": model_name,
                    "strategy": strategy,
                    "k": k_val,
                    "bal_acc": balanced_accuracy_score(y_te, y_pred),
                    "acc": accuracy_score(y_te, y_pred),
                    "prec": precision_score(y_te, y_pred, average='macro', zero_division=0),
                    "recall": recall_score(y_te, y_pred, average='macro', zero_division=0),
                    "f1": f1_score(y_te, y_pred, average='macro'),
                    "selected_indices": str(feat_idx.tolist())
                })

            pbar.update(1)

        pbar.close()

        # ------------------------------------------------------------
        # SAVE: detailed per fold
        # ------------------------------------------------------------
        detailed_path = os.path.join(out_dir, f"loo_nested_{strategy}_k{k_val}_detailed.csv")
        pd.DataFrame(rows).to_csv(detailed_path, index=False)

        # ------------------------------------------------------------
        # FEATURE FREQUENCIES
        # ------------------------------------------------------------
        merged_dfs = []
        for model_name, freq_counter in feat_freqs.items():
            df_freq = pd.DataFrame([
                {"feature_index": f, model_name: c} for f, c in freq_counter.items()
            ])
            merged_dfs.append(df_freq.set_index("feature_index"))

        merged = pd.concat(merged_dfs, axis=1).fillna(0)
        merged["count_total"] = merged.sum(axis=1)
        merged["frequency_%"] = (merged["count_total"] / (len(X) * len(model_spaces)) * 100).round(2)
        merged = merged.reset_index().sort_values("count_total", ascending=False)

        freq_path = os.path.join(out_dir, f"feature_frequency_{strategy}_k{k_val}_ALL.csv")
        merged.to_csv(freq_path, index=False)

        # ------------------------------------------------------------
        # SAVE SUMMARY (mean over folds)
        # ------------------------------------------------------------
        dsub = pd.DataFrame(rows)
        summary_rows = []

        for model_name, pred_data in preds_per_model.items():

            y_true_m = np.array(pred_data["y_true"])
            y_pred_m = np.array(pred_data["y_pred"])
            y_prob_m = np.array(pred_data["y_prob"])

            try:
                auc_m = roc_auc_score(y_true_m, y_prob_m, multi_class='ovr', average='macro')
            except:
                auc_m = np.nan

            dmodel = dsub[dsub["model"] == model_name]
            means = dmodel[["bal_acc", "acc", "prec", "recall", "f1"]].mean()
            stds = dmodel[["bal_acc", "acc", "prec", "recall", "f1"]].std()

            summary_rows.append({
                "model": model_name,
                "bal_acc_mean": means["bal_acc"],
                "acc_mean": means["acc"],
                "f1_mean": means["f1"],
                "auc": auc_m
            })

        summary = pd.DataFrame(summary_rows).round(3)
        summary = summary.sort_values("bal_acc_mean", ascending=False)

        print("\n📌 Mean performance over folds:")
        print(summary)

        summary.to_csv(os.path.join(out_dir, f"summary_{strategy}_k{k_val}.csv"), index=False)


# ============================================================
# 6️⃣ FINAL AGGREGATION + VISUALIZATION
# ============================================================

print("\n📊 Aggregating all results for final visualization...")

import matplotlib
matplotlib.use("Agg")
import seaborn as sns

# ------------------------------------------------------------
# LOAD ALL DETAILED FOLD-LEVEL RESULTS
# ------------------------------------------------------------
all_files = glob.glob(os.path.join(out_dir, "loo_nested_*_detailed.csv"))
all_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

# Ensure type consistency
all_df["k"] = all_df["k"].astype(int)

# ------------------------------------------------------------
# PLOT 1 — Balanced Accuracy vs k for each strategy
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=all_df,
    x="k",
    y="bal_acc",
    hue="strategy",
    estimator="mean",
    marker="o"
)
plt.title("Mean Balanced Accuracy vs Number of Features (k)")
plt.xlabel("Number of Selected Features (k)")
plt.ylabel("Balanced Accuracy (mean over folds)")
plt.legend(title="Strategy")
plt.tight_layout()
plot_path = os.path.join(out_dir, "PLOT_balanced_accuracy_vs_k.png")
plt.savefig(plot_path, dpi=300)
print(f"📁 Saved: {plot_path}")


# ------------------------------------------------------------
# LOAD ALL SUMMARY (MODEL-LEVEL) RESULTS
# ------------------------------------------------------------
summary_files = glob.glob(os.path.join(out_dir, "summary_*.csv"))
summary_df = pd.concat([pd.read_csv(f) for f in summary_files], ignore_index=True)

# Ensure numbers
summary_df["bal_acc_mean"] = summary_df["bal_acc_mean"].astype(float)
summary_df["auc"] = pd.to_numeric(summary_df["auc"], errors="coerce")
summary_df["k"] = summary_df["model"].map(lambda x: None)  # placeholder
summary_df["strategy"] = summary_df["model"].map(lambda x: None)  # placeholder

# Extract strategy + k from filename
for file in summary_files:
    fname = os.path.basename(file)
    parts = fname.split("_")      # summary_strategy_kX.csv
    strategy = parts[1]
    k = int(parts[2][1:].split(".")[0])
    tmp = pd.read_csv(file)
    tmp["strategy"] = strategy
    tmp["k"] = k
    summary_df = pd.concat([summary_df, tmp], ignore_index=True)

summary_df = summary_df.dropna(subset=["k"])

# ------------------------------------------------------------
# PLOT 2 — AUC vs k for each strategy
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=summary_df,
    x="k",
    y="auc",
    hue="strategy",
    marker="o"
)
plt.title("AUC vs Number of Features (k)")
plt.xlabel("Number of Selected Features (k)")
plt.ylabel("AUC (macro, OVR)")
plt.legend(title="Strategy")
plt.tight_layout()
plot_path = os.path.join(out_dir, "PLOT_auc_vs_k.png")
plt.savefig(plot_path, dpi=300)
print(f"📁 Saved: {plot_path}")


# ------------------------------------------------------------
# PLOT 3 — SELECT THE BEST MODEL OVERALL (based on AUC)
# ------------------------------------------------------------
best_idx = summary_df["auc"].idxmax()
best_model = summary_df.loc[best_idx]

print("\n🏆 BEST MODEL FOUND:")
print(best_model)

best_model.to_csv(os.path.join(out_dir, "BEST_MODEL.csv"), index=False)
print("📁 Saved: BEST_MODEL.csv")


# ------------------------------------------------------------
# PLOT 4 — FEATURE FREQUENCY HEATMAP
# ------------------------------------------------------------
freq_files = glob.glob(os.path.join(out_dir, "feature_frequency_*_ALL.csv"))
freq_df = pd.concat([pd.read_csv(f) for f in freq_files], ignore_index=True)

# Replace missing model columns with 0
for m in model_spaces.keys():
    if m not in freq_df.columns:
        freq_df[m] = 0

plt.figure(figsize=(12, 10))
freq_matrix = freq_df.pivot_table(
    values="count_total",
    index="feature_index",
    aggfunc="sum"
).fillna(0)

sns.heatmap(freq_matrix, cmap="viridis")
plt.title("Feature Selection Frequency Heatmap (all strategies + all k)")
plt.xlabel("Models")
plt.ylabel("Feature Index")
plt.tight_layout()
plot_path = os.path.join(out_dir, "PLOT_feature_frequency_heatmap.png")
plt.savefig(plot_path, dpi=300)
print(f"📁 Saved: {plot_path}")

print("\n🎉 All plots generated successfully!")
