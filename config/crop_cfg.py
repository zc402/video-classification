
# Index in "I" of IUV
lhand = [4]
rhand = [3]
larm = [21, 19, 17, 15]
rarm = [20, 22, 16, 18]
torso = [1, 2]
head = [23, 24]

# resize pixels
sm = 64
md = 128
lg = 256

properties = [
    (lhand, 'CropLHand', sm),
    (rhand, 'CropRHand', sm),

    # (larm, 'CropLArm', md),
    # (rarm, 'CropRArm', md),

    (torso + larm, 'CropTorsoLArm', lg),
    (torso + rarm, 'CropTorsoRArm', lg),
    # (head, 'CropHead'),
    (lhand + larm, 'CropLHandArm', md),
    (rhand + rarm, 'CropRHandArm', md),
    (lhand + larm + torso + head + rarm + rhand, 'CropHTAH', lg),
]

# {folder: size}
crop_part_args = [(x[0], x[1]) for x in properties]

"""
# crop_resize = {'CropLHand': 40, 'CropRHand': 40, 'CropHead': 40, 'CropTorso': 40,
# 'CropLArm': 100, 'CropRArm': 100, 'CropLHandArm': 100, 'CropRHandArm': 100, 'CropHeadTorso': 100,
# 'CropBody': 200, 'CropTorsoLArm': 200, 'CropTorsoRArm': 200}
# """
crop_resize_dict = {x[1]: x[2] for x in properties}

crop_folder_list = [x[1] for x in properties]