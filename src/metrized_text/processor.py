import time
from pathlib import Path

import cv2
import torch
import numpy as np
from torch.autograd import Variable
from metrized_text.craft import CRAFT

import craft_utils
import imgproc
import file_utils
from .utils import copy_state_dict


class TextDetector:
    def __init__(
        self,
        trained_model: str,
        test_folder: str,
        refiner_model: str = None,
        cuda: bool = False,
        canvas_size: int = 200,
        mag_ratio: float = 1.0,
        show_time: bool = False,
        poly: bool = False,
        link_threshold: float = 0.4,
        low_text: float = 0.1,
        text_threshold: float = 0.2,
        refine: bool = False,
    ):
        self.trained_model = trained_model
        self.test_folder = test_folder
        self.refiner_model = refiner_model
        self.cuda = cuda
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.show_time = show_time
        self.poly = poly
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.text_threshold = text_threshold
        self.refine = refine

        self.net = CRAFT()
        self.refine_net = None
        self.device = torch.device(
            "cuda" if self.cuda and torch.cuda.is_available() else "cpu"
        )
        self.result_folder = Path("result/maps")
        self.result_folder.mkdir(parents=True, exist_ok=True)

        self._load_model()

    def _load_model(self):
        print(f"Loading weights from checkpoint ({self.trained_model})")
        state = torch.load(self.trained_model, map_location=self.device)
        self.net.load_state_dict(copy_state_dict(state))
        self.net.to(self.device)
        self.net.eval()

        if self.refine:
            from refinenet import RefineNet

            self.refine_net = RefineNet()
            print(f"Loading refiner from checkpoint ({self.refiner_model})")
            state = torch.load(self.refiner_model, map_location=self.device)
            self.refine_net.load_state_dict(copy_state_dict(state))
            self.refine_net.to(self.device)
            self.refine_net.eval()
            self.poly = True

    def test_image(self, image_path):
        image = imgproc.loadImage(str(image_path))
        img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
            image,
            self.canvas_size,
            interpolation=cv2.INTER_LINEAR,
            mag_ratio=self.mag_ratio,
        )
        ratio_h = ratio_w = 1 / target_ratio

        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            y, feature = self.net(x)
            score_text = y[0, :, :, 0].cpu().numpy()
            score_link = y[0, :, :, 1].cpu().numpy()

            if self.refine_net:
                y_refiner = self.refine_net(y, feature)
                score_link = y_refiner[0, :, :, 0].cpu().numpy()

        boxes, polys, scores = craft_utils.getDetBoxes(
            score_text,
            score_link,
            self.text_threshold,
            self.link_threshold,
            self.low_text,
            self.poly,
        )

        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        polys = [p if p is not None else b for p, b in zip(polys, boxes)]

        return image, polys, score_text

    def process_folder(self):
        image_list, _, _ = file_utils.get_files(self.test_folder)
        start = time.time()

        for idx, image_path in enumerate(image_list):
            print(f"Processing {idx+1}/{len(image_list)}: {image_path}", end="\r")
            image, polys, score_text = self.test_image(image_path)

            out_path = self.result_folder / f"res_{Path(image_path).stem}_mask.jpg"
            cv2.imwrite(str(out_path), score_text)
            file_utils.saveResult(
                image_path, image[:, :, ::-1], polys, dirname=self.result_folder
            )

        print(f"\nTotal elapsed time: {time.time() - start:.2f}s")

    def detect_text_probability(self, image_path) -> float:
        image = imgproc.loadImage(str(image_path))
        img_resized, _, _ = imgproc.resize_aspect_ratio(
            image,
            self.canvas_size,
            interpolation=cv2.INTER_LINEAR,
            mag_ratio=self.mag_ratio,
        )
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            y, _ = self.net(x)
            score_text = y[0, :, :, 0].cpu().numpy()

        return float(np.clip(np.mean(score_text), 0.0, 1.0))
