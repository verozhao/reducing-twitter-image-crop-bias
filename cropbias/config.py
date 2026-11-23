"""Paths, constants and the fixed experiment settings.

Everything the pipeline writes stays inside the repository folder:
committed outputs go to data/ and results/, large local files go to work/,
and model caches go to .cache/.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK_SHARED = ROOT / "work"     # photos and raw query answers, shared by every run
# CROPBIAS_IMAGES points a run at another photo folder (for example the FairFace run).
IMAGES = Path(os.environ["CROPBIAS_IMAGES"]).resolve() if os.environ.get("CROPBIAS_IMAGES") else WORK_SHARED / "images"
# CROPBIAS_RUN_DIR puts one run (for example a small pilot) in its own folder.
_RUN = Path(os.environ["CROPBIAS_RUN_DIR"]).resolve() if os.environ.get("CROPBIAS_RUN_DIR") else None
DATA = (_RUN / "data") if _RUN else ROOT / "data"          # per-person tables (people, faces, pairs), kept local
RESULTS = (_RUN / "results") if _RUN else ROOT / "results"  # committed tables and figures
WORK = (_RUN / "work") if _RUN else WORK_SHARED             # composites, saliency maps, evaluation
CACHE = ROOT / ".cache"         # model weights and library caches
TWITTER_DIR = ROOT / "third_party" / "twitter"
TWITTER_BIN = TWITTER_DIR / "candidate_crops"
TWITTER_MODEL = TWITTER_DIR / "fastgaze.vxm"
TWITTER_RESULTS = TWITTER_DIR / "results.txt"


def setup_env() -> None:
    """Point every library cache at .cache/ so nothing is written outside the repo."""
    CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
    os.environ.setdefault("PIP_CACHE_DIR", str(CACHE / "pip"))


setup_env()

# Wikimedia asks automated clients to send a descriptive User-Agent.
USER_AGENT = os.environ.get(
    "CROPBIAS_USER_AGENT",
    "reducing-twitter-image-crop-bias/0.1 (academic research code)",
)

# ----------------------------------------------------------------------------
# Dataset recipe (Yee et al., CSCW 2021, Section 3.1, and Twitter's released
# notebook "Demographic Bias Analysis.ipynb").
# ----------------------------------------------------------------------------
GENDER_QIDS = {"Q6581072": "female", "Q6581097": "male"}
PORN_ACTOR_QID = "Q488111"

# Twitter's ethnic-group -> US census race map, copied from their notebook.
TWITTER_ETHNIC_RACE = {
    "Q127885": "white",   # Serbs
    "Q161652": "asian",   # Japanese
    "Q190168": "black",   # Yoruba people
    "Q42406": "white",    # English people
    "Q44806": "white",    # Ukrainians
    "Q49085": "black",    # African Americans
    "Q539051": "white",   # Greeks
    "Q678551": "white",   # American Jews
    "Q726673": "white",   # Swedish-speaking population of Finland
    "Q7325": "white",     # Jewish people
    "Q79797": "white",    # Armenians
    "Q179248": "white",   # Albanians
    "Q2325516": "white",  # Armenian Americans
}
ETHNIC_MIN_PEOPLE = 40      # ethnic groups with fewer people are dropped
SUBGROUP_MIN_PEOPLE = int(os.environ.get("CROPBIAS_MIN_SUBGROUP", 100))  # smaller race x gender subgroups are dropped
PHOTO_MIN_YEAR = 1950       # photos dated before this year are dropped
MIN_COLORFULNESS = 17.0     # Hasler-Suesstrunk colorfulness; below this a photo is black-and-white or sepia (set by eye; reported, not filtered)

GROUPS = ["BF", "BM", "WF", "WM"]
GROUP_LABELS = {"BF": "Black women", "BM": "Black men", "WF": "White women", "WM": "White men"}
GROUP_OF = {("black", "female"): "BF", ("black", "male"): "BM",
            ("white", "female"): "WF", ("white", "male"): "WM"}

# Six two-group comparisons. The first group is the one Twitter's audit found favored,
# so a positive gap means the same direction as the audit.
COMPARISONS = [("BF", "BM"), ("WF", "BF"), ("BF", "WM"), ("WF", "BM"), ("WM", "BM"), ("WF", "WM")]

# ----------------------------------------------------------------------------
# Images, faces and composites
# ----------------------------------------------------------------------------
DOWNLOAD_WIDTH = 500         # Commons thumbnail width (a standard thumbnail step)
FACE_MIN_PROB = 0.95         # MTCNN confidence for a face to count
FACE_SECOND_MAX_RATIO = 0.25 # a second face this large (area ratio) makes the photo ambiguous
FACE_MIN_HEIGHT_FRAC = 0.10  # face box must be at least this share of the photo height
COMPOSITE_HEIGHT = 256       # both photos are scaled to this height (Twitter's fixed-height setting)
JPEG_QUALITY = 95

# Crop aspect ratios as height / width, the convention of Twitter's crop program.
ASPECT_MAIN = 1.0            # square preview
ASPECT_ROBUST = 4.0 / 3.0    # 3:4 portrait preview (robustness check)
ASPECTS = {"square": ASPECT_MAIN, "portrait": ASPECT_ROBUST}
KEEP_FRACTION = 0.90         # a person is kept if at least 90% of the face box width is inside the crop
CUT_FRACTION = 0.10          # a face with more than 10% but less than 90% inside counts as cut

# Small changes used by stability fallback and by the predictability check.
PERTURB_SHIFT_FRAC = 0.02
PERTURB_BRIGHTNESS = 0.05
PERTURB_MAIN = ["mirror", "bright", "dark", "shift_r"]          # the proposal's four copies
PERTURB_GEOM = ["mirror", "shift_r", "shift_l", "mirror_shift"]  # no brightness change
VARIANTS = ["orig"] + sorted(set(PERTURB_MAIN) | set(PERTURB_GEOM))

# ----------------------------------------------------------------------------
# Design
# ----------------------------------------------------------------------------
SPLIT_SEED = 0
VAL_FRACTION = 0.30
PAIRS_PER_COMPARISON = {"val": int(os.environ.get("CROPBIAS_PAIRS_VAL", 1000)),
                        "test": int(os.environ.get("CROPBIAS_PAIRS_TEST", 4000))}
PAIR_SEED = {"val": 101, "test": 202}
DEEPGAZE_PER_COMPARISON = int(os.environ.get("CROPBIAS_DG_PER_COMP", 500))  # test composites per comparison scored by DeepGaze IIE
BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 7
