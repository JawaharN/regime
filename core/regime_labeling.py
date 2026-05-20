"""Map raw HMM state ids to intuitive labels.

Default strategy (`label_states`): rank states by mean realized return ascending,
then spread the ordered states across the configured label sequence. With the
default labels [crash, bear, neutral, bull, euphoria] and 3 fitted states we use
indices 0, 2, 4 → [crash, neutral, euphoria]. When BIC selects more states than
there are label names (n ∈ {6, 7} against the 5 default labels), the spread
re-uses adjacent labels — two states may share e.g. `bear` — rather than failing.

The final prompt also defines per-`n_components` label sets — exposed as
`LABEL_SETS` / `label_states_by_count` for human-facing displays. Labels never
drive position sizing: the orchestrator routes by *volatility rank*, not label.
"""

from __future__ import annotations

# Per-n_components human-facing label sets (worst → best), from the final prompt.
LABEL_SETS: dict[int, list[str]] = {
    3: ["BEAR", "NEUTRAL", "BULL"],
    4: ["CRASH", "BEAR", "BULL", "EUPHORIA"],
    5: ["CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"],
    6: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
    7: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "NEUTRAL", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
}


def label_states(state_returns: dict[int, float], label_names: list[str]) -> dict[int, str]:
    """state_returns: {state_id: mean_return}. Returns {state_id: label_name}.

    Lowest-mean state gets the worst label, highest-mean the best; intermediate
    states are spread evenly across the label sequence. When there are more
    states than labels, adjacent states share a label rather than failing.
    """
    if not state_returns:
        return {}

    n_states = len(state_returns)
    n_labels = len(label_names)
    if n_labels == 0:
        raise ValueError("label_names is empty")

    ordered = sorted(state_returns.items(), key=lambda kv: kv[1])

    if n_states == n_labels:
        chosen = list(range(n_labels))
    elif n_states == 1:
        chosen = [n_labels // 2]
    else:
        step = (n_labels - 1) / (n_states - 1)
        chosen = [min(n_labels - 1, round(i * step)) for i in range(n_states)]

    return {state_id: label_names[idx] for (state_id, _), idx in zip(ordered, chosen, strict=True)}


def label_states_by_count(state_returns: dict[int, float]) -> dict[int, str]:
    """Human-facing labels using the per-`n_components` set from the final prompt."""
    n = len(state_returns)
    labels = LABEL_SETS.get(n)
    if labels is None:
        # Fall back to the canonical 5-set spread.
        return label_states(state_returns, LABEL_SETS[5])
    ordered = sorted(state_returns.items(), key=lambda kv: kv[1])
    return {state_id: labels[i] for i, (state_id, _) in enumerate(ordered)}
