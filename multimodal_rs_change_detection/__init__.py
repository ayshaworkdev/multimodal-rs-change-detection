"""Utilities for multimodal remote-sensing change detection workflows."""

__all__ = [
    "CHANGE_CATEGORIES",
    "FEW_SHOT_EXAMPLES",
]

CHANGE_CATEGORIES = {"buildings", "roads", "parking lots", "vegetation", "water area"}

FEW_SHOT_EXAMPLES = {
    "no_change": [
        {"image_pair": "test_000001.png", "caption": "there is no difference."},
        {"image_pair": "test_000002.png", "caption": "the two scenes seem identical."},
        {"image_pair": "test_000003.png", "caption": "the scene is the same as before."},
        {"image_pair": "test_000005.png", "caption": "no change has occurred."},
        {"image_pair": "test_000006.png", "caption": "almost nothing has changed."},
    ],
    "change": [
        {"image_pair": "test_000166.png", "category": "buildings", "caption": "the trees have been replaced by many buildings."},
        {"image_pair": "test_000164.png", "category": "roads", "caption": "a road appears at the bottom and many houses are scattered replacing the trees."},
        {"image_pair": "test_000245.png", "category": "parking lots", "caption": "a parking lot at the top and a road in the center replaces many plants."},
        {"image_pair": "test_000004.png", "category": "vegetation", "caption": "the vegetation has been replaced by a road and many villas around."},
        {"image_pair": "test_water_area.png", "category": "water area", "caption": "a water area has been replaced by a road."},
    ],
}
