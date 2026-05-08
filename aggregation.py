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
MIN_RESPONSE_TOKENS = 6
MAX_RESPONSE_FRACTION = 0.50
MAX_RESPONSE_TOKENS = 192


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
        A 1-D feature tensor with one value per transition between transformer
        layers. For Qwen2.5-0.5B this is 23 values.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    n_hs = hidden_states.shape[0]
    n_layer_transitions = max(n_hs - 2, 0)

    features = hidden_states.new_zeros(n_layer_transitions)
    if n_layer_transitions == 0:
        return features.float()

    response_pos = _select_response_positions(hidden_states, attention_mask)
    n_response_tokens = response_pos.numel()
    if n_response_tokens <= 1:
        return features.float()

    # Dynamics between transformer blocks only: layer 2 - layer 1, ...,
    # final layer - previous layer. This excludes the embedding-to-layer-1 jump.
    contexts = hidden_states[2:, response_pos]
    updates = contexts - hidden_states[1:-1, response_pos]

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

    return js_scores.mean(dim=1).float()


def _select_response_positions(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Estimate response token positions from hidden-state dynamics.

    The exact assistant-response boundary is not available in the current
    ``aggregate(hidden_states, attention_mask)`` API because token ids and
    prompt length are not passed in. This uses a suffix prior plus a change
    point score over hidden-state geometry, which is less brittle than a fixed
    "last 64 tokens" slice while preserving the same external API.
    """
    token_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
    valid_pos = token_mask.nonzero(as_tuple=False).squeeze(-1)
    n_tokens = valid_pos.numel()
    if n_tokens <= MIN_RESPONSE_TOKENS:
        return valid_pos

    min_response = min(MIN_RESPONSE_TOKENS, max(n_tokens - 1, 1))
    max_by_fraction = max(TOKEN_TAILS, int(round(n_tokens * MAX_RESPONSE_FRACTION)))
    max_response = min(MAX_RESPONSE_TOKENS, max_by_fraction, n_tokens - 1)
    if max_response < min_response:
        return valid_pos[-min_response:]

    first_start = max(1, n_tokens - max_response)
    last_start = n_tokens - min_response
    candidate_starts = torch.arange(
        first_start,
        last_start + 1,
        device=hidden_states.device,
    )
    if candidate_starts.numel() == 0:
        return valid_pos[-min_response:]

    final_layer = hidden_states[-1, valid_pos].float()
    normed_final = F.normalize(final_layer, dim=-1)
    adjacent_shift = 1.0 - (normed_final[1:] * normed_final[:-1]).sum(dim=-1)
    boundary_score = adjacent_shift[candidate_starts - 1]

    if hidden_states.shape[0] >= 2:
        token_update = (
            hidden_states[-1, valid_pos].float() - hidden_states[-2, valid_pos].float()
        ).norm(dim=-1)
    else:
        token_update = final_layer.norm(dim=-1)

    cumsum = torch.cat([token_update.new_zeros(1), token_update.cumsum(dim=0)])
    prefix_mean = cumsum[candidate_starts] / candidate_starts.clamp_min(1)
    suffix_count = n_tokens - candidate_starts
    suffix_mean = (cumsum[-1] - cumsum[candidate_starts]) / suffix_count.clamp_min(1)
    dynamics_shift = (suffix_mean - prefix_mean).abs()

    response_lengths = (n_tokens - candidate_starts).float()
    length_prior = -(
        torch.log(response_lengths.clamp_min(1.0))
        - torch.log(response_lengths.new_tensor(float(TOKEN_TAILS)))
    ).abs()

    score = (
        _zscore(boundary_score)
        + 0.35 * _zscore(dynamics_shift)
        + 0.20 * _zscore(length_prior)
    )
    best_start = int(candidate_starts[torch.argmax(score)].item())
    response_pos = valid_pos[best_start:]

    # Most rows end with the literal <|endoftext|>, which tokenizes as EOS.
    # Token ids are unavailable here, so drop one trailing token only when the
    # selected response has enough content to make that conservative.
    if response_pos.numel() > min_response + 1:
        response_pos = response_pos[:-1]

    return response_pos


def _zscore(values: torch.Tensor) -> torch.Tensor:
    std = values.std(unbiased=False)
    if values.numel() <= 1 or std <= 1e-8:
        return torch.zeros_like(values)
    return (values - values.mean()) / std


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
        A 1-D float tensor. With the default aggregation for Qwen2.5-0.5B,
        this has 23 values, plus any optional geometric features.
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
