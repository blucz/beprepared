from nudenet import NudeDetector
from typing import List

from beprepared.node import Node
from beprepared.workspace import Workspace
from beprepared.dataset import Dataset
from beprepared.properties import CachedProperty

class NudeNetDetections:
    NUDITY_LABELS = {
        "BUTTOCKS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "ANUS_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
    }

    ALL_LABELS = {
        "FEMALE_GENITALIA_COVERED",
        "FACE_FEMALE",
        "BUTTOCKS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "ANUS_EXPOSED",
        "FEET_EXPOSED",
        "BELLY_COVERED",
        "FEET_COVERED",
        "ARMPITS_COVERED",
        "ARMPITS_EXPOSED",
        "FACE_MALE",
        "BELLY_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_COVERED",
        "FEMALE_BREAST_COVERED",
        "BUTTOCKS_COVERED",
    }

    def __init__(self, detections: List, threshold: float) -> None:
        self.threshold = threshold
        self.detections = detections

    def has_nudity(self):
        for detection in self.detections:
            if detection['class'] in self.NUDITY_LABELS and detection['score'] >= self.threshold:
                return True
        return False

    def has_female(self):
        for detection in self.detections:
            if detection['class'].startswith('MALE_') and detection['score'] >= self.threshold:
                return True

    def has_male(self):
        for detection in self.detections:
            if detection['class'].startswith('MALE_') and detection['score'] >= self.threshold:
                return True

    def has(self, label):
        for detection in self.detections:
            if detection['class'] == label and detection['score'] >= self.threshold:
                return True

    def show(self):
        segs = []
        if len(self.detections) == 0: 
            return "(nothing detected)"
        max_label_len = max(len(detection['class']) for detection in self.detections)
        segs.append(f"{'Label'.ljust(max_label_len)}  Score  BBox")
        for detection in self.detections:
            box = detection['box']
            segs.append(f"{detection['class'].ljust(max_label_len)}  {detection['score']:.2f}   {tuple(box)}")
        return '\n'.join(segs)

class NudeNet(Node):
    # TODO: support 640x640 model (see nudenet github)
    def __init__(self, threshold = 0.5):
        super().__init__()
        self.threshold = threshold

    def eval(self, dataset) -> Dataset:
        needs_detect = []

        for image in dataset.images:
            image.nudenet = CachedProperty('nudenet', image)
            if not image.nudenet.has_value:
                needs_detect.append(image)
            else:
                image.nudenet.value.threshold = self.threshold

        if len(needs_detect) >= 0:
            needs_detect_filenames = [self.workspace.get_path(image) for image in needs_detect]

            detector = NudeDetector()

            results = detector.detect_batch(needs_detect_filenames)
            for idx, image in enumerate(needs_detect):
                image.nudenet.value = NudeNetDetections(results[idx], self.threshold)
        for image in dataset.images:
            image.has_nudity = image.nudenet.value.has_nudity()

        return dataset
