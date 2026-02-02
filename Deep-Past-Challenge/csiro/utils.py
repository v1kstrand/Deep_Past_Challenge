from __future__ import annotations

import inspect


def filter_kwargs(func, kwargs: dict) -> dict:
    """
    Keep only keys that `func` can accept as keyword args.
    If func has **kwargs, keep everything.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {}

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)

    allowed = {
        name
        for name, p in params.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {k: v for k, v in kwargs.items() if k in allowed}
