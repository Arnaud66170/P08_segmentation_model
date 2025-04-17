# src/data_preprocessing/class_mapping.py

from typing import Dict, List

"""
Mapping des classes Cityscapes vers 8 superclasses simplifiées pour segmentation sémantique.

GROUPES :
0 - void         → à ignorer
1 - flat         → route, trottoir
2 - construction → bâtiments, murs, clôtures, etc.
3 - object       → poteaux, panneaux, feux tricolores
4 - nature       → arbres, herbe
5 - sky          → ciel
6 - human        → piétons, personnes
7 - vehicle      → toutes les formes de véhicules
"""
CLASS_MAPPING_P8 = {
    "void": [0, 1, 2, 3, 4, 5, 6],
    "flat": [7, 8, 9, 10],
    "construction": [11, 12, 13, 14, 15, 16],
    "object": [17, 18, 19, 20],
    "nature": [21, 22],
    "sky": [23],
    "human": [24, 25],
    "vehicle": [26, 27, 28, 29, 30, 31, 32, 33, -1]
}

# Mapping inverse : ID Cityscapes brut → classe indexée (0 à 7)
FLAT_CLASS_MAPPING: Dict[int, int] = {
    cityscapes_id: class_idx
    for class_idx, (_, id_list) in enumerate(CLASS_MAPPING_P8.items())
    for cityscapes_id in id_list
}

# Mapping ID de classe indexée → nom humain (pour visualisation)
CLASS_NAME_MAPPING: Dict[int, str] = {
    i: name for i, name in enumerate(CLASS_MAPPING_P8.keys())
}

# Optionnel : mapping nom classe → index (utile pour inférence/décodage inverse)
CLASS_INDEX_MAPPING: Dict[str, int] = {
    name: i for i, name in enumerate(CLASS_MAPPING_P8.keys())
}