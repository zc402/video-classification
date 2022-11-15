
# Index in "I" of IUV
lhand = [4]
rhand = [3]

lUArm = [15, 17]
rUArm = [16, 18]

lLArm = [19, 21]  # Left lower arm
rLArm = [20, 22]

larm = [21, 19, 17, 15]
rarm = [20, 22, 16, 18]
torso = [1, 2]
head = [23, 24]

# resize pixels
sm = 64
md = 128
lg = 192  # 256

properties = [

    (lhand + larm + torso + head + rarm + rhand, 'CropHTAH', lg),

    (lhand, 'CropLHand', sm),
    (rhand, 'CropRHand', sm),

    (lhand + larm, 'CropLHandArm', md),
    (rhand + rarm, 'CropRHandArm', md),

    (torso, 'CropTorso', md),

    # (lhand + lLArm, 'CropLHandLowArm', md),
    # (rhand + rLArm, 'CropRHandLowArm', md),

    # (larm, 'CropLArm', md),
    # (rarm, 'CropRArm', md),

    # (lhand + larm + torso, 'CropLHandArmTorso', lg), # lHAT
    # (rhand + rarm + torso, 'CropRHandArmTorso', lg),

    # (torso + lUArm + rUArm, 'CropToUpArm', md),
    # (torso + larm + rarm, 'CropToUpLoArm', md),
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