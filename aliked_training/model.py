"""Training adapter with the exact public ALIKED parameter names and shapes."""
from pathlib import Path
import torch
from .vendor.aliked import ALIKED
from .detector import TrainingDetector, integer_patches


class TrainableALIKED(ALIKED):
    def __init__(self, config, weights=None):
        super().__init__(model_name=config["model"], device="cpu", load_pretrained=False)
        self.dkd = TrainingDetector(radius=config["radius"], top_k=config["top_k"],
                                    random_k=config["random_k"], temperature=config["t_det"],
                                    peak_mode=config["peak_mode"], scores_th=config["eval_score_threshold"],
                                    n_limit=config["eval_max_keypoints"])
        # Upstream custom_ops CPU backward overwrites instead of accumulating
        # overlapping patches. Use the identical forward via indexed PyTorch.
        self.desc_head.get_patches_func = integer_patches
        if weights is not None:
            state = torch.load(Path(weights), map_location="cpu", weights_only=True)
            self.load_state_dict(state, strict=True)

    def forward(self, images, *, add_random=True, threshold=False):
        features, score_map = self.extract_dense_map(images)
        pred = self.dkd(score_map, add_random=add_random, threshold=threshold)
        # Upstream SDDH does not accept empty keypoint sets.
        descriptors, offsets = [], []
        for index, points in enumerate(pred["keypoints"]):
            if len(points):
                d, o = self.desc_head(features[index:index+1], [points])
                descriptors.append(d[0])
                offsets.append(o[0])
            else:
                descriptors.append(features.new_empty((0, features.shape[1])))
                offsets.append(features.new_empty((0, self.desc_head.n_pos, 2)))
        pred.update(descriptors=descriptors, offsets=offsets, score_map=score_map)
        return pred
