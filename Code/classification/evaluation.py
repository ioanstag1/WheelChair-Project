import os
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

# Scikit-learn imports
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION AND DATA LOADING
# ============================================================
# Input and Output paths
CSV_PATH = 'Combined_Features_CLEANED.csv'
OUT_DIR = 'Results/Brazil_Poland/Nested_CV_Results/'

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# Load dataset
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] Loaded dataset: {CSV_PATH}")
else:
    raise FileNotFoundError(f"[ERROR] Dataset not found: {CSV_PATH}")

# Prepare X (features) and y (labels)
y = df['label'].values
X = df.select_dtypes(include=[np.number]).drop(columns=['label'], errors='ignore')

print(f"[INFO] Samples: {len(X)}, Features: {X.shape[1]}")
print(f"[INFO] Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ============================================================
# 2. FEATURE SELECTION UTILITIES
# ============================================================

def _svm_weight_topk(Xtr, ytr, k, C=1.0):
    """Select top k features based on SVM weights."""
    Xs = StandardScaler().fit_transform(Xtr)
    clf = SVC(kernel='linear', C=C)
    clf.fit(Xs, ytr)
    # Average absolute weights across classes
    w = np.mean(np.abs(clf.coef_), axis=0)
    return np.argsort(w)[::-1][:min(k, Xtr.shape[1])]

def _ttest_topk(Xtr, ytr, k):
    """Select top k features based on ANOVA F-value."""
    sel = SelectKBest(score_func=f_classif, k=min(k, Xtr.shape[1]))
    sel.fit(Xtr, ytr)
    return np.where(sel.get_support())[0]

def _rfe_topk(Xtr, ytr, k):
    """Select top k features using Recursive Feature Elimination (RFE) with Linear SVM."""
    base = SVC(kernel='linear', C=1.0)
    Xs = StandardScaler().fit_transform(Xtr)
    sel = RFE(base, n_features_to_select=min(k, Xtr.shape[1]), step=1)
    sel.fit(Xs, ytr)
    return np.where(sel.support_)[0]

# ============================================================
# 3. CUSTOM SELECTION TRANSFORMER
# ============================================================

class StrategySelector(BaseEstimator, TransformerMixin):
    """Custom Scikit-learn transformer for dynamic feature selection strategies."""
    
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

        # Execute selection strategy
        if self.strategy == 'none':
            self.feature_indices_ = np.arange(X.shape[1])
        elif self.strategy == 'svm_w':
            self.feature_indices_ = _svm_weight_topk(X, y, k)
        elif self.strategy == 'ttest':
            self.feature_indices_ = _ttest_topk(X, y, k)
        elif self.strategy == 'rfe':
            self.feature_indices_ = _rfe_topk(X, y, k)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
            
        return self

    def transform(self, X):
        if not hasattr(X, 'columns'):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X.iloc[:, self.feature_indices_].values

# ============================================================
# 4. MODEL DEFINITIONS
# ============================================================
random_state = 42

model_spaces = {
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
            max_iter=2000, solver='liblinear', penalty='l1', class_weight='balanced', random_state=random_state
        ),
        "params": {"clf__C": [0.1, 1.0, 10.0]}
    },
    "RF": {
        "estimator": RandomForestClassifier(class_weight='balanced', random_state=random_state),
        "params": {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 4, 6]}
    }
}

# ============================================================
# 5. EXPERIMENT EXECUTION (NESTED CV)
# ============================================================
strategies = ['none','ttest', 'rfe', 'svm_w'] # Define desired strategies here
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
outer_cv = LeaveOneOut()

for strategy in strategies:
    print(f"\n========================================")
    print(f"[INFO] Running Strategy: {strategy}")
    print(f"========================================")

    # Define range of features to test (k)
    k_values = [X.shape[1]] if strategy == "none" else range(5, X.shape[1] + 1)

    for k_val in k_values:
        print(f"\n[INFO] Evaluating k={k_val} features...")

        rows = []
        preds_per_model = {m: {"y_true": [], "y_prob": [], "y_pred": []} for m in model_spaces.keys()}
        feat_freqs = {m: Counter() for m in model_spaces.keys()}

        # Progress bar for Leave-One-Out loop
        pbar = tqdm(total=len(X), desc=f"{strategy} | k={k_val}", ncols=100)

        # Outer Loop (Leave-One-Out)
        for i, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), 1):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            for model_name, spec in model_spaces.items():
                
                # Build pipeline: Scaling -> Selection -> Classifier
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("select", StrategySelector(strategy=strategy, k=k_val)),
                    ("clf", spec["estimator"])
                ])

                # Inner Loop (Grid Search for Hyperparameter Tuning)
                grid = {"select__k": [k_val]}
                for p, vals in spec["params"].items():
                    grid[p] = vals

                gs = GridSearchCV(
                    pipe, [grid],
                    scoring="balanced_accuracy",
                    cv=inner_cv,
                    n_jobs=-1,
                    refit=True
                )
                
                gs.fit(X_tr, y_tr)
                best_model = gs.best_estimator_

                # Predict on held-out test sample
                y_pred = best_model.predict(X_te)
                
                # Get probabilities for AUC (handle cases where proba is unavailable)
                try:
                    y_prob = best_model.predict_proba(X_te)[0]
                except:
                    n_classes = len(np.unique(y))
                    y_prob = np.ones(n_classes) / n_classes

                # Track selected features
                feat_idx = best_model.named_steps["select"].feature_indices_
                feat_freqs[model_name].update(feat_idx)

                # Store predictions
                preds_per_model[model_name]["y_true"].append(int(y_te[0]))
                preds_per_model[model_name]["y_pred"].append(int(y_pred[0]))
                preds_per_model[model_name]["y_prob"].append(y_prob)

                # Store iteration results
                rows.append({
                    "fold": i,
                    "model": model_name,
                    "strategy": strategy,
                    "k": k_val,
                    "bal_acc": balanced_accuracy_score(y_te, y_pred),
                    "selected_indices": str(feat_idx.tolist())
                })

            pbar.update(1)
        pbar.close()

        # ------------------------------------------------------------
        # SAVE RESULTS
        # ------------------------------------------------------------
        
        # 1. Detailed per-fold results
        detailed_path = os.path.join(OUT_DIR, f"loo_nested_{strategy}_k{k_val}_detailed.csv")
        pd.DataFrame(rows).to_csv(detailed_path, index=False)

        # 2. Feature selection frequency
        merged_dfs = []
        for model_name, freq_counter in feat_freqs.items():
            df_freq = pd.DataFrame([
                {"feature_index": f, model_name: c} for f, c in freq_counter.items()
            ])
            merged_dfs.append(df_freq.set_index("feature_index"))

        merged = pd.concat(merged_dfs, axis=1).fillna(0)
        merged["count_total"] = merged.sum(axis=1)
        freq_path = os.path.join(OUT_DIR, f"feature_frequency_{strategy}_k{k_val}_ALL.csv")
        merged.reset_index().to_csv(freq_path, index=False)

        # 3. Summary Performance (Balanced Accuracy & AUC)
        summary_rows = []
        for model_name, pred_data in preds_per_model.items():
            y_true_m = np.array(pred_data["y_true"])
            y_prob_m = np.array(pred_data["y_prob"])
            
            # Calculate AUC (Macro Average for Multiclass)
            try:
                auc_m = roc_auc_score(y_true_m, y_prob_m, multi_class='ovr', average='macro')
            except ValueError:
                auc_m = np.nan

            # Calculate mean Balanced Accuracy across folds
            dmodel = pd.DataFrame(rows)
            dmodel = dmodel[dmodel["model"] == model_name]
            mean_bal_acc = dmodel["bal_acc"].mean()

            summary_rows.append({
                "model": model_name,
                "bal_acc_mean": mean_bal_acc,
                "auc": auc_m
            })

        summary_df = pd.DataFrame(summary_rows).round(3)
        summary_df = summary_df.sort_values("bal_acc_mean", ascending=False)

        print("\n[INFO] Performance Summary:")
        print(summary_df)

        summary_path = os.path.join(OUT_DIR, f"summary_{strategy}_k{k_val}.csv")
        summary_df.to_csv(summary_path, index=False)

print("\n[SUCCESS] Nested Cross-Validation completed successfully.")