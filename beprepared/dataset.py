class Dataset:
    def __init__(self, images=None):
        self.images = images or []

    def copy(self):
        return Dataset([image.copy() for image in self.images])
