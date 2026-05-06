"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

TOKEN_TAILS = 64


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    n_hs = hidden_states.shape[0]
    n_layers = n_hs - 1

    features = hidden_states.new_zeros(n_layers)
    if n_layers <= 1:
        return features.float()

    # suppose that last TOKEN_TAILS tokens are the response
    token_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
    response_pos = token_mask.nonzero(as_tuple=False).squeeze(-1)[-TOKEN_TAILS:]
    n_response_tokens = response_pos.numel()
    if n_response_tokens <= 1:
        return features.float()

    # delta = x^l - x^(l-1) for each token on each layer
    contexts = hidden_states[1:n_layers, response_pos]
    updates = contexts - hidden_states[: n_layers - 1, response_pos]

    # compute update directions of hidden states and project onto them
    projections = torch.einsum("lcd,ltd->lct", contexts, updates)
    projections = projections / contexts.norm(dim=-1).clamp_min(1e-8).unsqueeze(-1)

    # Jensen-Shannon divergence between the projection and uniform distribution
    log_proj = F.log_softmax(projections, dim=1)
    proj = log_proj.exp()
    uniform = 1.0 / n_response_tokens
    log_uniform = -torch.log(
        projections.new_tensor(n_response_tokens, dtype=torch.float32)
    )
    midpoint = 0.5 * (proj + uniform)
    log_midpoint = midpoint.log()

    js_scores = 0.5 * (proj * (log_proj - log_midpoint)).sum(dim=1) + 0.5 * (
        uniform * (log_uniform - log_midpoint)
    ).sum(dim=1)

    features[1:n_layers] = js_scores.mean(dim=1)
    return features.float()


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    # Placeholder: returns an empty tensor (no geometric features).
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
