import torch as th
import numpy as np
from PIL import Image
# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
# pip install opencv-python
import cv2

class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, size=224, framerate=-1, ):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def video_to_tensor(self, video_file, preprocess, total_duration, start_time=None, end_time=None):
        # Samples a frame sample_fp X frames.
        cap = cv2.VideoCapture(video_file)
        # frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps = int(cap.get(cv2.CAP_PROP_FPS))

        start_time = start_time if start_time is not None else 0.0
        end_time = end_time if end_time is not None else total_duration
        ret, frame = cap.read()
        images = []

        while ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))
            ret, frame = cap.read()

        cap.release()

        if len(images) > 0:
            n = len(images)
            start_idx = int(start_time*n/total_duration)
            end_idx = int(end_time*n/total_duration)
            video_data = th.tensor(np.stack(images[start_idx: end_idx]))
        else:
            video_data = th.zeros(1)
        return video_data

    def get_video_data(self, video_path, duration, start_time=None, end_time=None):
        image_input = self.video_to_tensor(video_path, self.transform, duration, start_time=start_time, end_time=end_time)
        return image_input

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

# An ordinary video frame extractor based CV2
RawVideoExtractor = RawVideoExtractorCV2
