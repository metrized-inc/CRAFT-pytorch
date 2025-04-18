import time
from pathlib import Path

import cv2
import torch
import numpy as np
from torch.autograd import Variable
from craft import CRAFT

import craft_utils
import imgproc
import file_utils
from .utils import copy_state_dict


class TextDetector:
    def __init__(self, args: dict):
        self.args = args
        self.net = CRAFT()
        self.refine_net = None
        self.device = torch.device(
            "cuda" if args["cuda"] and torch.cuda.is_available() else "cpu"
        )
        self.result_folder = Path(__file__).parent / "result" / "maps"
        self.result_folder.mkdir(parents=True, exist_ok=True)

        self._load_model()

    def _load_model(self):
        print(f"Loading weights from checkpoint ({self.args['trained_model']})")
        state = torch.load(self.args["trained_model"], map_location=self.device)
        self.net.load_state_dict(copy_state_dict(state))
        self.net.to(self.device)
        self.net.eval()

        if self.args["refine"]:
            from refinenet import RefineNet

            self.refine_net = RefineNet()
            print(f"Loading refiner from ({self.args['refiner_model']})")
            state = torch.load(self.args["refiner_model"], map_location=self.device)
            self.refine_net.load_state_dict(copy_state_dict(state))
            self.refine_net.to(self.device)
            self.refine_net.eval()
            self.args["poly"] = True

    def test_image(self, image_path):
        image = imgproc.loadImage(str(image_path))
        img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
            image,
            self.args["canvas_size"],
            interpolation=cv2.INTER_LINEAR,
            mag_ratio=self.args["mag_ratio"],
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
            self.args["text_threshold"],
            self.args["link_threshold"],
            self.args["low_text"],
            self.args["poly"],
        )

        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        polys = [p if p is not None else b for p, b in zip(polys, boxes)]

        return image, polys, score_text

    def process_folder(self):
        image_list, _, _ = file_utils.get_files(self.args["test_folder"])
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
