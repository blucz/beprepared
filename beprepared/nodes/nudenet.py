from nudenet import NudeDetector
from typing import List
from tqdm import tqdm

from beprepared.node import Node
from beprepared.workspace import Workspace
from beprepared.dataset import Dataset
from beprepared.properties import CachedProperty, ComputedProperty

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
    '''The NudeNet node is used to detect nudity in images. 

    This can be used to filter out adult content. It can also be used to avoid sending NSFW content to censored models. For example, you may elect to
    use GPT4o to caption SFW images, but JoyCaption to caption NSFW images.

    Nudenet is quick and not very resource intensive, but it achieves pretty accurate results.'''

    # TODO: support 640x640 model (see nudenet github)
    def __init__(self, threshold = 0.5):
        '''Initializes the NudeNet node

        Args:
            threshold (float): The minimum confidence score for a label to be considered positive (default is 0.5)
        '''
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

            batch_size = 128
            for i in tqdm(range(0, len(needs_detect_filenames), batch_size), desc="NudeNet"):
                batch = needs_detect_filenames[i:i + batch_size]
                results = detector.detect_batch(batch)
                for idx, image in enumerate(needs_detect[i:i + batch_size]):
                    image.nudenet.value = NudeNetDetections(results[idx], self.threshold)

        for image in dataset.images:
            image.has_nudity = ComputedProperty(lambda image: image.nudenet.value.has_nudity() if image.nudenet.has_value else None)

        return dataset

__all__ = ['NudeNet']
