"""Single source of truth for the main tab order.

Every piece of code that switches/inspects the current main tab MUST import
these constants instead of hardcoding integers — reordering the tabs has
already silently broken the right-panel mapping once (KZ 4→5 shift when the
FS Explorer tab landed).

Order chosen by the user: analysis views first (BM, FS, FS Explorer, KZ,
Results), tooling after.
"""

IDX_BM = 0
IDX_FS = 1
IDX_FS_EXPLORER = 2
IDX_KZ = 3
IDX_RESULTS = 4
IDX_MDC = 5
IDX_NOTES = 6
IDX_HELP = 7
IDX_START = 8

TAB_TITLES = [
    "BM", "FS", "FS Explorer", "KZ", "Results",
    "MDC Fit", "Notes", "Help", "Start",
]
