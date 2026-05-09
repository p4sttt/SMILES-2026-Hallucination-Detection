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
RESPONSE_FRACTION = 0.35
MAX_RESPONSE_TOKENS = 192
FEATURES_PER_TRANSITION = 10
GLOBAL_FEATURES = 8


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
        A compact 1-D feature tensor with ICR-like projection statistics,
        hidden-state dynamics summaries, and a few global response/context
        alignment features. For Qwen2.5-0.5B this is
        ``23 * FEATURES_PER_TRANSITION + GLOBAL_FEATURES`` values.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    n_hs = hidden_states.shape[0]
    n_layer_transitions = max(n_hs - 2, 0)

    features = hidden_states.new_zeros(
        n_layer_transitions * FEATURES_PER_TRANSITION + GLOBAL_FEATURES
    )
    if n_layer_transitions == 0:
        return features.float()

    context_pos = _real_token_positions(hidden_states, attention_mask)
    response_pos = _select_response_positions(hidden_states, attention_mask)
    n_context_tokens = context_pos.numel()
    n_response_tokens = response_pos.numel()
    if n_context_tokens <= 1 or n_response_tokens <= 1:
        return features.float()

    # Dynamics between transformer blocks only: layer 2 - layer 1, ...,
    # final layer - previous layer. This excludes the embedding-to-layer-1 jump.
    response_prev = hidden_states[1:-1, response_pos].float()
    response_next = hidden_states[2:, response_pos].float()
    response_updates = response_next - response_prev

    # ICR-like projection: i ranges over response tokens, while j ranges over
    # all real prompt+response tokens. This preserves the prompt-context signal.
    context_basis = hidden_states[1:-1, context_pos].float()
    context_basis = context_basis / context_basis.norm(dim=-1).clamp_min(1e-8).unsqueeze(
        -1
    )
    projections = torch.einsum("lrd,lcd->lrc", response_updates, context_basis)

    log_proj = F.log_softmax(projections, dim=-1)
    proj = log_proj.exp()
    uniform = 1.0 / n_context_tokens
    log_uniform = -torch.log(
        projections.new_tensor(n_context_tokens, dtype=torch.float32)
    )
    midpoint = 0.5 * (proj + uniform)
    log_midpoint = midpoint.log()

    js_scores = 0.5 * (proj * (log_proj - log_midpoint)).sum(dim=-1) + 0.5 * (
        uniform * (log_uniform - log_midpoint)
    ).sum(dim=-1)
    js_mean = js_scores.mean(dim=1)

    entropy = -(proj * log_proj).sum(dim=-1)
    entropy = entropy / torch.log(projections.new_tensor(float(n_context_tokens)))
    entropy_mean = entropy.mean(dim=1)

    max_prob = proj.max(dim=-1).values.mean(dim=1)
    top_k = min(5, n_context_tokens)
    topk_mass = proj.topk(top_k, dim=-1).values.sum(dim=-1).mean(dim=1)

    # response_pos is a suffix of context_pos in the current API-compatible
    # heuristic, so the last n_response_tokens columns correspond to response.
    response_mass = proj[..., -n_response_tokens:].sum(dim=-1).mean(dim=1)

    update_norm_by_token = response_updates.norm(dim=-1)
    update_norm_mean = update_norm_by_token.mean(dim=1)
    update_norm_std = update_norm_by_token.std(dim=1, unbiased=False)

    cosine_drift = 1.0 - F.cosine_similarity(response_next, response_prev, dim=-1)
    cosine_drift_mean = cosine_drift.mean(dim=1)
    cosine_drift_std = cosine_drift.std(dim=1, unbiased=False)

    hidden_norm = response_next.norm(dim=-1).mean(dim=1)

    icr_features = torch.stack(
        [
            js_mean,
            entropy_mean,
            max_prob,
            topk_mass,
            response_mass,
            update_norm_mean,
            update_norm_std,
            cosine_drift_mean,
            cosine_drift_std,
            hidden_norm,
        ],
        dim=1,
    ).flatten().float()

    global_features = _global_response_features(
        hidden_states,
        context_pos,
        response_pos,
        update_norm_by_token,
    )
    return torch.cat([icr_features, global_features], dim=0).float()


def _global_response_features(
    hidden_states: torch.Tensor,
    context_pos: torch.Tensor,
    response_pos: torch.Tensor,
    update_norm_by_token: torch.Tensor,
) -> torch.Tensor:
    n_context_tokens = context_pos.numel()
    n_response_tokens = response_pos.numel()

    final_response = hidden_states[-1, response_pos].float()
    final_norm = final_response.norm(dim=-1)
    final_response_mean = final_response.mean(dim=0)
    last_response = final_response[-1]

    prompt_pos = context_pos[:-n_response_tokens]
    if prompt_pos.numel() > 0:
        prompt_mean = hidden_states[-1, prompt_pos].float().mean(dim=0)
        prompt_response_cosine = F.cosine_similarity(
            final_response_mean.unsqueeze(0),
            prompt_mean.unsqueeze(0),
            dim=-1,
        ).squeeze(0)
    else:
        prompt_response_cosine = final_response.new_tensor(0.0)

    last_to_mean_cosine = F.cosine_similarity(
        last_response.unsqueeze(0),
        final_response_mean.unsqueeze(0),
        dim=-1,
    ).squeeze(0)

    final_update_norm = update_norm_by_token[-1]
    return torch.stack(
        [
            torch.log1p(final_response.new_tensor(float(n_context_tokens))),
            final_response.new_tensor(float(n_response_tokens / n_context_tokens)),
            final_norm.mean(),
            final_norm.std(unbiased=False),
            update_norm_by_token.mean(),
            final_update_norm.mean(),
            prompt_response_cosine,
            last_to_mean_cosine,
        ]
    ).float()


def _real_token_positions(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    token_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
    return token_mask.nonzero(as_tuple=False).squeeze(-1)


def _select_response_positions(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Select likely assistant-response positions with a simple suffix prior.

    The exact assistant-response boundary is not available in the current
    ``aggregate(hidden_states, attention_mask)`` API because token ids, text,
    and prompt length are not passed in. Keep this conservative and stable:
    treat the answer as a suffix, with a slightly adaptive length instead of a
    hard last-64 slice.
    """
    valid_pos = _real_token_positions(hidden_states, attention_mask)
    n_tokens = valid_pos.numel()
    if n_tokens <= MIN_RESPONSE_TOKENS:
        return valid_pos

    adaptive_tail = max(TOKEN_TAILS, int(round(n_tokens * RESPONSE_FRACTION)))
    n_response = min(
        max(MIN_RESPONSE_TOKENS, adaptive_tail),
        MAX_RESPONSE_TOKENS,
        n_tokens,
    )
    return valid_pos[-n_response:]


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
        this has 238 values, plus any optional geometric features.
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
