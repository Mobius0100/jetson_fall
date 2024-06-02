import cv2
import os, math
import logging
import numpy as np
import time

from queue import Queue
from threading import Thread, Lock
from pathlib import Path
from threading import Thread
# from augment import LetterBox

# from YOLOV8_infer import YOLOV8
# from utils import draw_one_picture, make_outdir

LOGGER = logging.getLogger("FallDetect")


class CamLoader_Q:
    """Use threading and queue to capture a frame and store to queue for pickup in sequence.
    Recommend for video file.

    Args:
        camera: (int, str) Source of camera or video.,
        batch_size: (int) Number of batch frame to store in queue. Default: 1,
        queue_size: (int) Maximum queue size. Default: 256,
        preprocess: (Callable function) to process the frame before return.
    """
    def __init__(self, camera, batch_size=1, queue_size=256, preprocess=None):
        self.stream = cv2.VideoCapture(camera)
        assert self.stream.isOpened(), 'Cannot read camera source!'
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.frame_size = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # Queue for storing each frames.

        self.stopped = False
        self.batch_size = batch_size
        self.Q = Queue(maxsize=queue_size)

        self.preprocess_fn = preprocess

    def start(self):
        t = Thread(target=self.update, args=(), daemon=True).start()
        c = 0
        while not self.grabbed():
            time.sleep(0.1)
            c += 1
            if c > 20:
                self.stop()
                raise TimeoutError('Can not get a frame from camera!!!')
        return self

    def update(self):
        while not self.stopped:
            if not self.Q.full():
                frames = []
                for k in range(self.batch_size):
                    ret, frame = self.stream.read()
                    if not ret:
                        self.stop()
                        return

                    if self.preprocess_fn is not None:
                        frame = self.preprocess_fn(frame)

                    frames.append(frame)
                    frames = np.stack(frames)
                    self.Q.put(frames)
            else:
                with self.Q.mutex:
                    self.Q.queue.clear()
            # time.sleep(0.05)

    def grabbed(self):
        """Return `True` if can read a frame."""
        return self.Q.qsize() > 0

    def getitem(self):
        return self.Q.get().squeeze()

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        self.stream.release()

    def __len__(self):
        return self.Q.qsize()

    def __del__(self):
        if self.stream.isOpened():
            self.stream.release()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream.isOpened():
            self.stream.release()


class LoadStream(object):
    """
        read video file or camera(index: 0,1,2.. , rtsp)
    """

    def __init__(self, sources='./data/test.mp4', vid_stride=1, transforms=None) -> None:
        self.vid_stride = vid_stride
        self.transforms = transforms

        sources = [sources]
        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        for i, s in enumerate(sources):
            cap = cv2.VideoCapture(s)
            if not cap.isOpened():
                raise ConnectionError(f'Failed to open {s}')
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30

            success, self.imgs[i] = cap.read()
            if not success or self.imgs[i] is None:
                raise ConnectionError(f'Failed to read images from {s}')
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f'Success! frame of shape {w}x{h} at {self.fps[i]:.2f} FPS')
            self.threads[i].start()
    
    def update(self, i, cap, stream):
        """ Read stream i frames in daemon thread"""
        n, f = 0, self.frames[i]
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()
            if n % self.vid_stride == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING! Video stream unresponsive, please check you carmera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)
            time.sleep(0.0)

    def get_count(self):
        return self.frames
    
    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads):
            # cv2.destroyAllWindows()
            raise StopIteration

        im0 = self.imgs.copy()
        im = [x for x in im0]
        # if self.transforms:
        #     im = np.stack([self.transforms(x) for x in im0])  # transforms
        # else:
        #     im = np.stack([LetterBox()(image=x) for x in im0])
        #     im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        #     im = np.ascontiguousarray(im)  # contiguous

        return "0", im[0], im0

class CamLoader:
    """Use threading to capture a frame from camera for faster frame load.
    Recommend for camera or webcam.

    Args:
        camera: (int, str) Source of camera or video.,
        preprocess: (Callable function) to process the frame before return.
    """
    def __init__(self, camera, preprocess=None, ori_return=False):
        self.stream = cv2.VideoCapture(camera)
        assert self.stream.isOpened(), 'Cannot read camera source!'
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.frame_size = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.stopped = False
        self.ret = False
        self.frame = None
        self.ori_frame = None
        self.read_lock = Lock()
        self.ori = ori_return

        self.preprocess_fn = preprocess

    def start(self):
        self.t = Thread(target=self.update, args=())  # , daemon=True)
        self.t.start()
        c = 0
        while not self.ret:
            time.sleep(0.1)
            c += 1
            if c > 20:
                self.stop()
                raise TimeoutError('Can not get a frame from camera!!!')
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            self.read_lock.acquire()
            self.ori_frame = frame.copy()
            if ret and self.preprocess_fn is not None:
                frame = self.preprocess_fn(frame)

            self.ret, self.frame = ret, frame
            self.read_lock.release()

    def grabbed(self):
        """Return `True` if can read a frame."""
        return self.ret

    def getitem(self):
        self.read_lock.acquire()
        frame = self.frame.copy()
        ori_frame = self.ori_frame.copy()
        self.read_lock.release()
        if self.ori:
            return frame, ori_frame
        else:
            return frame

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        if self.t.is_alive():
            self.t.join()
        self.stream.release()

    def __del__(self):
        if self.stream.isOpened():
            self.stream.release()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream.isOpened():
            self.stream.release()

def main():
    pass
    # stream = LoadStream("rtsp://admin:Admin123456@192.168.1.51:554/Streaming/Channels/101")
    # # stream = CamLoader("rtsp://admin:Admin123456@192.168.1.51:554/Streaming/Channels/101").start()
    # stream = CamLoader_Q("./data/test_dance.mp4").start()
    # model = YOLOV8('./data/pose.bmodel')
    # # stream = LoadStream("./data/pose.mp4")
    # # stream_iter = iter(stream)
    # out_path = "./data/predit/output"
    # out_path = make_outdir(out_path)

    # i = 1
    # while stream.grabbed():
    #     # sources, img, img0 = next(stream_iter)
    #     img = stream.getitem()

    #     t1 = time.time()
    #     preprocessed_image, original_image, scale = model.preprocess_image(img, 640)
    #     t2 = time.time()
    #     outputs = model.predit(preprocessed_image)
    #     t3 = time.time()
    #     results = model.postprocess_pose(outputs)
    #     t4 = time.time()
        
    #     print(f'one picture cost {t4-t1}ms. pre: {t2-t1}ms, infer: {t3-t2}ms, post: {t4-t3}ms.')

    #     out_img = draw_one_picture(original_image, results, scale)

    #     cv2.imwrite(f"{out_path}/{i}.jpg", out_img)
    #     print(i)

    #     i += 1
    
    # stream.stop()

if __name__ == "__main__":
    main()