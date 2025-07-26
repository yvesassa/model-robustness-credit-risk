# model_robustness_eval.py
"""
Module: model_robustness_eval

Provides functions to evaluate model robustness under various perturbations and adversarial attacks.
Includes:
 - Covariate perturbation (gaussian noise, feature shift, random masking)
 - Featurewise perturbation
 - Gradient-based L0 attack
 - Loss-sensitive L0 attack
 - FGSM L2 attack
 - PGD L2 attack
 - FGSM Linf attack
 - PGD Linf attack
 - Generic evaluation and plotting functions
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score

# =============================================================================
# === Covariate Perturbation Functions ===
# =============================================================================
def evaluate_covariate_perturbation(
    X_test,
    y_test,
    model,
    methods=['gaussian_noise'],
    noise_level=0.05,
    mask_prob=0.1,
    threshold=0.5,
    featurewise_std=False,
    feature_names=None,
    random_state=42,
    verbose=True,
    return_metrics=False
):
    """
    Apply global covariate perturbations and evaluate model performance.
    """
    rng = np.random.default_rng(random_state)
    Xp = X_test.copy()
    X_vals = Xp.values if hasattr(Xp, 'values') else Xp
    y_vals = y_test.values if hasattr(y_test, 'values') else y_test

    # apply each specified method sequentially
    for method in methods:
        if method == 'gaussian_noise':
            if featurewise_std and isinstance(Xp, pd.DataFrame):
                noise = np.zeros_like(X_vals)
                for i, col in enumerate(Xp.columns):
                    std = Xp[col].std()
                    noise[:, i] = rng.normal(0, noise_level * std, size=len(Xp))
            else:
                noise = rng.normal(0, noise_level, size=X_vals.shape)
            X_vals = X_vals + noise

        elif method == 'feature_shift':
            X_vals = X_vals * (1 + noise_level)

        elif method == 'random_mask':
            mask = rng.binomial(1, 1 - mask_prob, size=X_vals.shape)
            X_vals = X_vals * mask

        else:
            raise ValueError(f"Unknown perturbation method: {method}")

    # predictions
    y_pred_proba = model.predict_proba(X_vals)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    if verbose:
        print(f"\nCovariate Perturbation: {methods}")
        print(f" - noise_level: {noise_level}, mask_prob: {mask_prob}")
        print(f" - threshold: {threshold}")
        print(classification_report(y_vals, y_pred, target_names=['Class 0', 'Class 1']))
        print(f"AUC-ROC: {roc_auc_score(y_vals, y_pred_proba):.4f}")

    # prediction drift
    original_pred = model.predict(X_test.values if hasattr(X_test, 'values') else X_test)
    changes = np.sum(original_pred != y_pred)
    change_pct = changes / len(y_test) * 100
    if verbose:
        print(f"Prediction changes: {changes}/{len(y_test)} ({change_pct:.2f}%)")

    metrics = {
        "auc": roc_auc_score(y_vals, y_pred_proba),
        "accuracy": accuracy_score(y_vals, y_pred),
        "f1": f1_score(y_vals, y_pred),
        "pred_change_count": changes,
        "pred_change_pct": change_pct
    }

    if return_metrics:
        return metrics, X_vals
    return X_vals

# =============================================================================
# === Featurewise Perturbation Functions ===
# =============================================================================
def perturb_single_feature(
    X: pd.DataFrame,
    feature: str,
    method: str = 'gaussian_noise',
    noise_level: float = 0.05,
    mask_prob: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Perturb a single feature column in X by chosen method.
    """
    rng = np.random.default_rng(random_state)
    X_perturbed = X.copy()

    if method == 'gaussian_noise':
        std = X[feature].std()
        noise = rng.normal(0, noise_level * std, size=X.shape[0])
        X_perturbed[feature] += noise

    elif method == 'feature_shift':
        shift = noise_level * X[feature].std()
        X_perturbed[feature] += shift

    elif method == 'random_mask':
        mask = rng.binomial(1, 1 - mask_prob, size=X.shape[0])
        X_perturbed[feature] *= mask

    else:
        raise ValueError(f"Unknown method: {method}")

    return X_perturbed


def evaluate_featurewise_perturbation(
    X_test: pd.DataFrame,
    y_test,
    model,
    method='gaussian_noise',
    noise_level=0.05,
    mask_prob=0.1
) -> pd.DataFrame:
    """
    Evaluate model performance after perturbing each feature individually.
    Returns DataFrame of impact metrics.
    """
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame with .columns")

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # baseline
    y_pred_base = model.predict(X_test.values)
    y_prob_base = model.predict_proba(X_test.values)[:, 1]
    baseline_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_base),
        "f1": f1_score(y_test, y_pred_base),
        "auc": roc_auc_score(y_test, y_prob_base)
    }

    feature_impact = []
    for feature in X_test.columns:
        X_pert = perturb_single_feature(
            X_test, feature, method=method, noise_level=noise_level, mask_prob=mask_prob
        )
        y_prob_pert = model.predict_proba(X_pert.values)[:, 1]
        y_pred_pert = (y_prob_pert >= 0.5).astype(int)

        delta_acc = baseline_metrics['accuracy'] - accuracy_score(y_test, y_pred_pert)
        delta_f1 = baseline_metrics['f1'] - f1_score(y_test, y_pred_pert)
        delta_auc = baseline_metrics['auc'] - roc_auc_score(y_test, y_prob_pert)
        pred_change_pct = (y_pred_base != y_pred_pert).mean() * 100

        feature_impact.append({
            "feature": feature,
            "delta_accuracy": delta_acc,
            "delta_f1": delta_f1,
            "delta_auc": delta_auc,
            "pred_change_pct": pred_change_pct
        })
    df_impact = pd.DataFrame(feature_impact)
    df_impact.sort_values('delta_auc', ascending=False, inplace=True)
    return df_impact


def plot_top_features(
    df_impact: pd.DataFrame,
    metric: str = 'delta_auc',
    top_n: int = 10,
    title: str = None
) -> None:
    """
    Plot top_n features by chosen impact metric.
    """
    df_top = df_impact.set_index('feature')[metric].sort_values(ascending=True).tail(top_n)
    df_top.plot(kind='barh', figsize=(8, 6))
    plt.xlabel(f"Change in {metric.upper()}")
    plt.title(title or f"Top {top_n} Features by {metric.upper()}")
    plt.tight_layout()
    plt.show()

# =============================================================================
# === Gradient-Based L0 Attack ===
# =============================================================================
def gradient_topk_l0_attack(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.1,
    max_features: int = 5,
    feature_names=None
):
    """
    Gradient-based L0 attack; perturbs top-k features per sample.
    Returns adversarial X and saliency_log list.
    """
    device = next(model.parameters()).device
    model.eval()

    X_tensor = torch.tensor(X.astype(np.float32), requires_grad=True, device=device)
    y_tensor = torch.tensor(y.astype(np.int64), device=device)

    logits, _ = model(X_tensor)
    loss = torch.nn.functional.cross_entropy(logits, y_tensor)
    loss.backward()
    grads = X_tensor.grad.detach().cpu().numpy()

    X_adv = X.copy()
    saliency_log = []

    for i in range(X.shape[0]):
        grad_i = grads[i]
        topk = np.argsort(np.abs(grad_i))[-max_features:]
        top_features = (
            [feature_names[j] for j in topk]
            if feature_names else topk.tolist()
        )
        saliency_log.append(top_features)
        for j in topk:
            X_adv[i, j] += epsilon * np.sign(grad_i[j])
    return X_adv, saliency_log

# =============================================================================
# === Loss-Sensitive L0 Attack ===
# =============================================================================
def loss_sensitive_l0_attack(
    model,
    X_df: pd.DataFrame,
    y: np.ndarray,
    max_features: int = 5,
    epsilon: float = 0.1,
    batch_size: int = 512
):
    """
    Loss-sensitive L0 attack; uses loss differences to select features.
    Returns adversarial X and saliency_log.
    """
    X_vals = X_df.values
    feature_names = list(X_df.columns)
    n_samples, n_features = X_vals.shape

    # original loss
    eps = 1e-8
    y_proba_orig = model.predict_proba(X_vals)[:, 1]
    base_loss = -(y * np.log(y_proba_orig + eps) + (1 - y) * np.log(1 - y_proba_orig + eps))

    X_adv = X_vals.copy()
    saliency_log = []

    for i in range(n_samples):
        x_i = X_vals[i].reshape(1, -1)
        losses = []
        for j in range(n_features):
            x_rep = np.tile(x_i, (n_features, 1))
            noise = np.random.normal(0, epsilon, size=n_features)
            x_rep[:, j] += noise
            pert_probs = model.predict_proba(x_rep)[:, 1]
            pert_loss = -(y[i] * np.log(pert_probs + eps) + (1 - y[i]) * np.log(1 - pert_probs + eps))
            losses.append(pert_loss - base_loss[i])
        losses = np.array(losses)
        topk = np.argsort(losses)[-max_features:]
        saliency_log.append([feature_names[j] for j in topk])
        for j in topk:
            X_adv[i, j] += np.random.normal(0, epsilon)
    return X_adv, saliency_log

# =============================================================================
# === FGSM L2 Attack ===
# =============================================================================
def fgsm_l2_attack(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 3.0
) -> np.ndarray:
    """
    Fast Gradient Sign Method (L2 norm) attack.
    """
    model.eval()
    X_tensor = torch.tensor(X.astype(np.float32), requires_grad=True)
    y_tensor = torch.tensor(y, dtype=torch.long)

    logits, _ = model(X_tensor)
    loss = torch.nn.functional.cross_entropy(logits, y_tensor)
    loss.backward()
    grad = X_tensor.grad.detach().cpu().numpy()

    # normalize to unit L2 per sample
    grad_norm = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-8
    perturb = epsilon * grad / grad_norm
    X_adv = X + perturb
    return X_adv

# =============================================================================
# === PGD L2 Attack ===
# =============================================================================
def pgd_l2_attack(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 3.0,
    alpha: float = 0.5,
    iters: int = 10
) -> np.ndarray:
    """
    Projected Gradient Descent (L2 norm) attack.
    """
    model.eval()
    X_tensor = torch.tensor(X.astype(np.float32))
    y_tensor = torch.tensor(y, dtype=torch.long)
    X_adv = X_tensor.clone().detach().requires_grad_(True)

    for _ in range(iters):
        logits, _ = model(X_adv)
        loss = torch.nn.functional.cross_entropy(logits, y_tensor)
        loss.backward()
        grad = X_adv.grad.detach().cpu().numpy()

        # normalize gradient
        grad_norm = np.linalg.norm(grad, ord=2, axis=1, keepdims=True) + 1e-8
        grad_unit = grad / grad_norm
        X_adv = X_adv + alpha * torch.tensor(grad_unit, dtype=torch.float32)

        # project back
        delta = X_adv.detach().cpu().numpy() - X_tensor.detach().cpu().numpy()
        delta_norm = np.linalg.norm(delta, ord=2, axis=1, keepdims=True)
        factor = np.clip(epsilon / delta_norm, a_min=1.0, a_max=None)
        X_adv = torch.tensor(X_tensor.detach().cpu().numpy() + delta * factor, dtype=torch.float32)
        X_adv.requires_grad_(True)

    return X_adv.detach().cpu().numpy()

# =============================================================================
# === FGSM Linf Attack ===
# =============================================================================
def fgsm_linf_attack(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.1
) -> np.ndarray:
    """
    Fast Gradient Sign Method (Linf norm) attack.
    """
    model.eval()
    X_tensor = torch.tensor(X.astype(np.float32), requires_grad=True)
    y_tensor = torch.tensor(y.astype(np.int64))
    logits, _ = model(X_tensor)
    loss = torch.nn.functional.cross_entropy(logits, y_tensor)
    loss.backward()
    grad = X_tensor.grad.detach()

    perturb = epsilon * torch.sign(grad)
    X_adv = X_tensor + perturb
    return X_adv.detach().cpu().numpy()

# =============================================================================
# === PGD Linf Attack ===
# =============================================================================
def pgd_linf_attack(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.1,
    alpha: float = 0.02,
    iters: int = 10
) -> np.ndarray:
    """
    Projected Gradient Descent (Linf norm) attack.
    """
    model.eval()
    X_tensor = torch.tensor(X.astype(np.float32))
    y_tensor = torch.tensor(y.astype(np.int64))
    X_adv = X_tensor.clone().detach().requires_grad_(True)

    for _ in range(iters):
        logits, _ = model(X_adv)
        loss = torch.nn.functional.cross_entropy(logits, y_tensor)
        loss.backward()
        grad = X_adv.grad.detach()
        X_adv = X_adv + alpha * torch.sign(grad)

        # clip to Linf ball
        X_adv = torch.max(
            torch.min(X_adv, X_tensor + epsilon),
            X_tensor - epsilon
        )
        X_adv = X_adv.detach().requires_grad_(True)

    return X_adv.detach().cpu().numpy()

# =============================================================================
# === Generic Attack Evaluation ===
# =============================================================================
def evaluate_attack(
    X_adv: np.ndarray,
    y_true: np.ndarray,
    model,
    label: str = "Attack",
    threshold: float = 0.5
) -> dict:
    """
    Evaluate model on adversarial examples.
    Prints metrics and returns dict.
    """
    y_proba = model.predict_proba(X_adv)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n[{label}] Evaluation")
    print(f"AUC:        {auc:.4f}")
    print(f"Accuracy:   {acc:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Class 0", "Class 1"]))

    return {"auc": auc, "accuracy": acc, "f1": f1}

# =============================================================================
# === Usage Example (as comments) ===
# =============================================================================
# import pandas as pd
# from model_robustness_eval import (
#     evaluate_covariate_perturbation,
#     evaluate_featurewise_perturbation, plot_top_features,
#     gradient_topk_l0_attack, loss_sensitive_l0_attack,
#     fgsm_l2_attack, pgd_l2_attack,
#     fgsm_linf_attack, pgd_linf_attack,
#     evaluate_attack
# )
#
# # Load your test data
# X_test_df = pd.read_csv('X_test.csv')\#
# y_test = pd.read_csv('y_test.csv')['target']
# model = ...  # your trained sklearn or torch model
#
# # Covariate perturbation
# evaluate_covariate_perturbation(X_test_df, y_test, model, methods=['gaussian_noise', 'random_mask'], noise_level=0.05)
#
# # Featurewise impact
# df_imp = evaluate_featurewise_perturbation(X_test_df, y_test, model, method='feature_shift', noise_level=0.1)
# plot_top_features(df_imp, metric='delta_auc')
#
# # L0 attacks
# X_np = X_test_df.values
# y_np = y_test.values
# X_grad, grad_feats = gradient_topk_l0_attack(model, X_np, y_np, epsilon=0.1)
# evaluate_attack(X_grad, y_np, model, label="Gradient L0")
#
# X_loss, loss_feats = loss_sensitive_l0_attack(model, X_test_df, y_np)
# evaluate_attack(X_loss, y_np, model, label="Loss-Based L0")
#
# # L2 attacks
# X_fgsm = fgsm_l2_attack(model, X_np, y_np, epsilon=3.0)
# evaluate_attack(X_fgsm, y_np, model, label="FGSM L2")
#
# X_pgd = pgd_l2_attack(model, X_np, y_np, epsilon=3.0, alpha=0.5, iters=10)
# evaluate_attack(X_pgd, y_np, model, label="PGD L2")
#
# # Linf attacks
# X_fgsm_inf = fgsm_linf_attack(model, X_np, y_np, epsilon=0.1)
# evaluate_attack(X_fgsm_inf, y_np, model, label="FGSM Linf")
#
# X_pgd_inf = pgd_linf_attack(model, X_np, y_np, epsilon=0.1, alpha=0.02, iters=10)
# evaluate_attack(X_pgd_inf, y_np, model, label="PGD Linf")
