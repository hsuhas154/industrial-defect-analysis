"""idefect: Industrial Surface Defect Analysis.

A modular, reproducible prototype that compares a classical computer-vision
pipeline (preprocessing + HOG features + Decision Tree / SVM) against a small
custom CNN on the NEU Surface Defect Database.
"""

__version__ = "0.1.0"

# Canonical six NEU classes and the two-letter filename prefixes used by the
# flat NEU-CLS distribution.
NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

NEU_PREFIX_TO_CLASS = {
    "Cr": "crazing",
    "In": "inclusion",
    "Pa": "patches",
    "PS": "pitted_surface",
    "RS": "rolled-in_scale",
    "Sc": "scratches",
}
