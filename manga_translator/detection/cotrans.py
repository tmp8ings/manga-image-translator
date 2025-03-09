import os
import shutil
import numpy as np
import torch
import cv2
import einops
from typing import List, Tuple

from .default_utils.DBNet_resnet34 import TextDetection as TextDetectionDefault
from .default_utils import imgproc, dbnet_utils, craft_utils
from .common import OfflineDetector
from ..utils import TextBlock, Quadrilateral, det_rearrange_forward

MODEL = None
def det_batch_forward_default(batch: np.ndarray, device: str):
    global MODEL
    if isinstance(batch, list):
        batch = np.array(batch)
    batch = einops.rearrange(batch.astype(np.float32) / 127.5 - 1.0, 'n h w c -> n c h w')
    batch = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        db, mask = MODEL(batch)
        db = db.sigmoid().cpu().numpy()
        mask = mask.cpu().numpy()
    return db, mask

class CotransDetector(OfflineDetector):
    _MODEL_MAPPING = {
        'model': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/detect.ckpt',
            'hash': '69080aea78de0803092bc6b751ae283ca463011de5f07e1d20e6491b05571a30',
            'file': '.',
        }
    }

    def __init__(self, *args, **kwargs):
        os.makedirs(self.model_dir, exist_ok=True)
        if os.path.exists('detect.ckpt'):
            shutil.move('detect.ckpt', self._get_file_path('detect.ckpt'))
        super().__init__(*args, **kwargs)

    async def _load(self, device: str):
        self.model = TextDetectionDefault()
        sd = torch.load(self._get_file_path('detect.ckpt'), map_location='cpu')
        self.model.load_state_dict(sd['model'] if 'model' in sd else sd)
        self.model.eval()
        self.device = device
        if device == 'cuda' or device == 'mps':
            self.model = self.model.to(self.device)
        global MODEL
        MODEL = self.model

    async def _unload(self):
        del self.model

    async def _infer(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                     unclip_ratio: float, verbose: bool = False):
        # Preprocess image (similar to run_detection)
        img_filtered = cv2.bilateralFilter(image, 17, 80, 80)
        img_resized = img_filtered.astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(einops.rearrange(img_resized, 'h w c -> 1 c h w')).to(self.device)
        # Forward pass
        with torch.no_grad():
            db, mask = self.model(img_tensor)
            db = db.sigmoid().cpu().numpy()
            mask = mask[0, 0, :, :].cpu().numpy()
        # Use original image shape for detection
        h, w = image.shape[:2]
        det = dbnet_utils.SegDetectorRepresenter(text_threshold, box_threshold, unclip_ratio=unclip_ratio)
        boxes, scores = det({'shape': [(h, w)]}, db)
        boxes, scores = boxes[0], scores[0]
        if boxes.size == 0:
            polys = []
        else:
            idx = boxes.reshape(boxes.shape[0], -1).sum(axis=1) > 0
            polys, _ = boxes[idx], scores[idx]
            polys = polys.astype(np.float64)
            polys = craft_utils.adjustResultCoordinates(polys, 1, 1, ratio_net=1)
            polys = polys.astype(np.int64)
        textlines = [Quadrilateral(pts.astype(int), '', score) for pts, score in zip(polys, scores)]
        textlines = list(filter(lambda q: q.area > 16, textlines))
        # Resize mask as in reference (doubling dimensions)
        mask_resized = cv2.resize(mask, (mask.shape[1] * 2, mask.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
        raw_mask = np.clip(mask_resized * 255, 0, 255).astype(np.uint8)
        return textlines, raw_mask, None
