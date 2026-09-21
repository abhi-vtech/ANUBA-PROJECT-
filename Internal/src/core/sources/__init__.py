"""Frame sources: one small adapter per runtime, all yielding `Frame`.

Each adapter's only job is to turn that runtime's native output into
`src.core.contract.Frame` and to map class ids to label strings.  None of them
contains analysis logic -- that is the point of the split.

They are imported lazily by `src.core.runner` so that importing the core on a
machine without torch (or without pyservicemaker) still works.
"""
