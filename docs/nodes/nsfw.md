# NSFW

## NudeNet

[NudeNet](https://github.com/notAI-tech/NudeNet) uses a fine-tuned yolov8 model to detect concepts related to SFW/NSFW distinctions in images.

### Parameters

- `threshold` (default: 0.5): The minimum confidence required to recognize a concept in the image.

### Output properties

- `image.has_nudity`: Whether the image contains nudity or not.
- `image.nudenet`: The raw output of the NudeNet model, of type `NudeNetDetections`
- `image.nudenet.value.detections`: A list of dicts like `{ "class": "<label>", "score": 0.8 }` which contain the detected concepts and their confidence scores.
- `image.nudenet.value.has_nudity`: Whether the image contains nudity or not.
- `image.nudenet.value.has_female`: Whether the image contains a female subject.
- `image.nudenet.value.has_male`: Whether the image contains a male subject.
- `image.nudenet.value.has(label)`: Whether the score for a specified label exceeds the threshold.

### Supported Labels

    FEMALE_GENITALIA_COVERED
    FACE_FEMALE
    BUTTOCKS_EXPOSED
    FEMALE_BREAST_EXPOSED
    FEMALE_GENITALIA_EXPOSED
    MALE_BREAST_EXPOSED
    ANUS_EXPOSED
    FEET_EXPOSED
    BELLY_COVERED
    FEET_COVERED
    ARMPITS_COVERED
    ARMPITS_EXPOSED
    FACE_MALE
    BELLY_EXPOSED
    MALE_GENITALIA_EXPOSED
    ANUS_COVERED
    FEMALE_BREAST_COVERED
    BUTTOCKS_COVERED

### Example

```python
dataset >> NudeNet
```
