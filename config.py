DEPTH_RANGE = 3.0
DEPTH_MIN = -1.5

stb_joints = [
    'loc_bn_palm_L',
    'loc_bn_pinky_L_01',
    'loc_bn_pinky_L_02',
    'loc_bn_pinky_L_03',
    'loc_bn_pinky_L_04',
    'loc_bn_ring_L_01',
    'loc_bn_ring_L_02',
    'loc_bn_ring_L_03',
    'loc_bn_ring_L_04',
    'loc_bn_mid_L_01',
    'loc_bn_mid_L_02',
    'loc_bn_mid_L_03',
    'loc_bn_mid_L_04',
    'loc_bn_index_L_01',
    'loc_bn_index_L_02',
    'loc_bn_index_L_03',
    'loc_bn_index_L_04',
    'loc_bn_thumb_L_01',
    'loc_bn_thumb_L_02',
    'loc_bn_thumb_L_03',
    'loc_bn_thumb_L_04',
]

rhd_joints = [
    'loc_bn_palm_L',
    'loc_bn_thumb_L_04',
    'loc_bn_thumb_L_03',
    'loc_bn_thumb_L_02',
    'loc_bn_thumb_L_01',
    'loc_bn_index_L_04',
    'loc_bn_index_L_03',
    'loc_bn_index_L_02',
    'loc_bn_index_L_01',
    'loc_bn_mid_L_04',
    'loc_bn_mid_L_03',
    'loc_bn_mid_L_02',
    'loc_bn_mid_L_01',
    'loc_bn_ring_L_04',
    'loc_bn_ring_L_03',
    'loc_bn_ring_L_02',
    'loc_bn_ring_L_01',
    'loc_bn_pinky_L_04',
    'loc_bn_pinky_L_03',
    'loc_bn_pinky_L_02',
    'loc_bn_pinky_L_01'
]

snap_joint_names = [
    'loc_bn_palm_L',
    'loc_bn_thumb_L_01',
    'loc_bn_thumb_L_02',
    'loc_bn_thumb_L_03',
    'loc_bn_thumb_L_04',
    'loc_bn_index_L_01',
    'loc_bn_index_L_02',
    'loc_bn_index_L_03',
    'loc_bn_index_L_04',
    'loc_bn_mid_L_01',
    'loc_bn_mid_L_02',
    'loc_bn_mid_L_03',
    'loc_bn_mid_L_04',
    'loc_bn_ring_L_01',
    'loc_bn_ring_L_02',
    'loc_bn_ring_L_03',
    'loc_bn_ring_L_04',
    'loc_bn_pinky_L_01',
    'loc_bn_pinky_L_02',
    'loc_bn_pinky_L_03',
    'loc_bn_pinky_L_04'
]

SNAP_BONES = [
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20)
]

SNAP_PARENT = [
    0,  # 0's parent
    0,  # 1's parent
    1,
    2,
    3,
    0,  # 5's parent
    5,
    6,
    7,
    0,  # 9's parent
    9,
    10,
    11,
    0,  # 13's parent
    13,
    14,
    15,
    0,  # 17's parent
    17,
    18,
    19,
]

JOINT_COLORS = (
    (216, 31, 53),
    (214, 208, 0),
    (136, 72, 152),
    (126, 199, 216),
    (0, 0, 230),
)

DEFAULT_CACHE_DIR = 'datasets/data/.cache'

USEFUL_BONE = [1, 2, 3,
               5, 6, 7,
               9, 10, 11,
               13, 14, 15,
               17, 18, 19]

kinematic_tree = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]

ID2ROT = {
        2: 13, 3: 14, 4: 15,
        6: 1, 7: 2, 8: 3,
        10: 4, 11: 5, 12: 6,
        14: 10, 15: 11, 16: 12,
        18: 7, 19: 8, 20: 9,
    }