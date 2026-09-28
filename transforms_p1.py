"""Phase-1 transforms for Mobile-CDNet (Opus 5.5).
Input to every transform: image = uint8/float HxWx6 **BGR** [pre(0:3) | post(3:6)] as read by cv2 in dataset.py, label HxW.
Fix vs. original Transforms.ToTensor: the original reversed all 6 channels ([:, :, ::-1]), which produced
[post_RGB | pre_RGB], i.e. the model's `pre` input was actually the post image. ToTensorRGB flips BGR->RGB per image.
"""
import random
import numpy as np
import cv2


class PerTemporalColorJitter(object):
    """Independent photometric jitter for pre and post (different acquisition dates / radiometry). Applied on uint8 BGR."""
    def __init__(self, p=0.8, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03):
        self.p, self.b, self.c, self.s, self.h = p, brightness, contrast, saturation, hue

    def _jit(self, bgr):
        x = bgr.astype(np.float32)
        ops = [0, 1, 2, 3]; random.shuffle(ops)
        for op in ops:
            if op == 0:    # brightness
                x = x * random.uniform(1 - self.b, 1 + self.b)
            elif op == 1:  # contrast
                m = x.mean(); x = (x - m) * random.uniform(1 - self.c, 1 + self.c) + m
            elif op == 2:  # saturation
                g = x.mean(2, keepdims=True); x = (x - g) * random.uniform(1 - self.s, 1 + self.s) + g
            else:          # hue (OpenCV H in [0,180))
                hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[..., 0] = (hsv[..., 0] + random.uniform(-self.h, self.h) * 180) % 180
                x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        return np.clip(x, 0, 255).astype(np.uint8)

    def __call__(self, image, label):
        image = image.copy()
        if random.random() < self.p: image[:, :, 0:3] = self._jit(image[:, :, 0:3])
        if random.random() < self.p: image[:, :, 3:6] = self._jit(image[:, :, 3:6])
        return [image, label]


class RandomPreShift(object):
    """Shift ONLY the pre image by up to +-max_shift px (label is aligned to post). Simulates the measured
    BCDD pre/post misregistration (median ~3px, p75 ~5.6px)."""
    def __init__(self, max_shift=4, p=0.5):
        self.m, self.p = max_shift, p

    def __call__(self, image, label):
        if random.random() < self.p:
            h, w = image.shape[:2]
            dx, dy = random.randint(-self.m, self.m), random.randint(-self.m, self.m)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            image = image.copy()
            image[:, :, 0:3] = cv2.warpAffine(np.ascontiguousarray(image[:, :, 0:3]), M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return [image, label]


class RandomRot90(object):
    def __call__(self, image, label):
        k = random.randint(0, 3)
        if k:
            image = np.ascontiguousarray(np.rot90(image, k, axes=(0, 1)))
            label = np.ascontiguousarray(np.rot90(label, k, axes=(0, 1)))
        return [image, label]


class ToTensorRGB(object):
    """BGR->RGB per temporal image, keeps order [pre | post]."""
    def __call__(self, image, label):
        import torch
        image = image[:, :, [2, 1, 0, 5, 4, 3]].transpose((2, 0, 1)).copy()
        return [torch.from_numpy(image), torch.LongTensor(np.array(label, dtype=int)).unsqueeze(dim=0)]
