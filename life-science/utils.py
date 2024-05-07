import torch
from monai.transforms import MapTransform


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi-channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor), and ET (Enhancing tumor).

    Reference: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb

    """

    def __call__(self, data):
        data_dict = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct Tumor Core
            result.append(torch.logical_or(data_dict[key] == 2, data_dict[key] == 3))
            # merge labels 1, 2 and 3 to construct Whole Tumor
            result.append(
                torch.logical_or(
                    torch.logical_or(data_dict[key] == 2, data_dict[key] == 3), data_dict[key] == 1
                )
            )
            # label 2 is Enhancing Tumor
            result.append(data_dict[key] == 2)
            data_dict[key] = torch.stack(result, axis=0).float()
        return data_dict