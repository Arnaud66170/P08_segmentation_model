# src/data_preprocessing/class_mapping.py
# Mapping simplifié pour les 8 grandes classes Cityscapes
# (à adapter selon les IDs exacts du dataset officiel)

CLASS_MAPPING_P8 = {
    0: 0,   # road
    1: 1,   # sidewalk
    2: 2,   # building
    3: 3,   # wall
    4: 4,   # fence
    5: 5,   # pole
    6: 6,   # traffic light
    7: 7    # traffic sign
    # tous les autres IDs = ignorés
}
