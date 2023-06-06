def dice_coef_multiclass(inputs, targets, num_class, smooth=1):
    dice = 0
    for index in range(num_class):
        dice += dice_coef(inputs[:, index, ...], targets[:, index, ...], smooth)
    return 1 - dice / num_class

def dice_coef(inputs, targets, smooth=1):
    inputs = inputs.flatten()
    targets = targets.flatten()
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return dice
