from typing import Any


def tqdm_wrap(iterable: Any, desc: str | None = None, total: int | None = None, leave: bool = False):
    try:
        from tqdm.auto import tqdm
    except Exception as exc:
        raise ImportError("tqdm_wrap requires tqdm. Install with `pip install tqdm`.") from exc
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=leave,
        dynamic_ncols=True,
        mininterval=0.1,
    )
