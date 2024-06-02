import time
import argparse
import yaml
import cv2
import numpy as np
import onnxruntime as ort
import torch

import functools
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import common
# from data_process import PreprocessYOLO

TRT_LOGGER = trt.Logger()
np.set_printoptions(suppress=True)

def get_engine(trt_file):
    print(f"Read engine from file {trt_file}.")
    with open(trt_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read()) 
    


class YOLOV8POSE:
    def __init__(self, model_path, confidence_thres=0.5, iou_thres=0.5, model_shape=(640, 640), mode='trt') -> None:

        self.model_path = model_path
        self.conf = float(confidence_thres)
        self.iou = float(iou_thres)
        self.mode = mode
        self.model_shape = model_shape

        with open('coco128.yaml', 'r') as f:
            self.classes = yaml.safe_load(f)['names']

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        if self.mode == 'trt':
            self.init_tensorrt()

    def init_tensorrt(self):

        with get_engine(self.model_path) as engine, engine.create_execution_context() as self.context:

            self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(engine)
            # data = np.random.rand(1,3,640,640)
            # print(f'data type: {data.dtype}, {type(data)}')
            self.inputs[0].host = np.random.rand(1, 3, 640, 640).astype(np.float32)
            trt_outputs = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
            # trt_outputs = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)

    
        print('warm engine file...')


    def draw_detections(self, img, box, score, class_id, keypoints):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # plot keypoints

        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]

        # draw point
        for x, y, conf in keypoints:
            if conf > self.conf:
                cv2.circle(img, (int(x), int(y)), radius=10, color=(0, 0, 255), thickness=-1)

        # draw line
        for conn in connections:
            kp1 = keypoints[conn[0]]
            kp2 = keypoints[conn[1]]

            conf1 = kp1[2]
            conf2 = kp2[2]

            if conf1 > self.conf and conf2 > self.conf:
                x1, y1 = kp1[:2]
                x2, y2 = kp2[:2]

                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)



    def preprocess(self, input_img):

        if isinstance(input_img, str):
            self.img = cv2.imread(input_img)
        else:
            self.img = input_img

        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.model_shape[0], self.model_shape[1]))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32).copy()   # use copy create a new data.

        # Return the preprocessed image data
        return image_data
    

    def postprocess(self, input_img, outputs):

        if len(outputs[0].shape) <= 3:
            outputs[0] = outputs[0].reshape(-1, 56, 8400)
        
        outputs = outputs[0]
        
        outputs = outputs[0, :, outputs[0, 4, :] > self.conf]
        rows = outputs.shape[0]

        x_factor = self.img_width / self.model_shape[0]
        y_factor = self.img_height / self.model_shape[1]


        class_ids = []
        scores = []
        boxes = []
        xyxy_boxes = []
        keypoints = []
        norm_keypoints = []
            

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4]

            max_score = float(classes_scores)
            # If the maximum score is above the confidence threshold
            if max_score >= self.conf:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                # print(f'xywh: {x}, {y}, {w}, {h}')
                xyxy_box = [
                    x - (0.5 * w),
                    y - (0.5 * h),
                    x + (0.5 * w),
                    y + (0.5 * h)
                ]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                keypoint = outputs[i][5:].tolist()
                kpts = []
                norm_kpts = []

                for i in range(0, len(keypoint), 3):
                    x = (keypoint[i] - self.img_width / 2) / (self.img_width / 2)
                    y = (keypoint[i+1] - self.img_height / 2) / (self.img_height / 2)
                    keypoint[i] = keypoint[i] * x_factor
                    keypoint[i+1] = keypoint[i+1] * y_factor
                    
                    # print(keypoint[i:i+3])
                    kpts.append(keypoint[i:i+3])
                    norm_kpts.append([x, y, keypoint[i+2]]) 

                # keypoint = [keypoint[i:i+3] for i in range(0, len(keypoint), 3)]

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
                xyxy_boxes.append(xyxy_box)
                keypoints.append(kpts)
                norm_keypoints.append(norm_kpts)

        # print(len(xyxy_boxes))
        indices = cv2.dnn.NMSBoxes(xyxy_boxes, scores, self.conf, self.iou)
        # print(len(indices))

        if len(indices) != 0:
            indices = indices.flatten()

        self.pose_data = []

        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            keypoint = keypoints[i]
            norm_keypoint = norm_keypoints[i]
            xyxy_box = xyxy_boxes[i]
            kpt_data = {
                'box' : xyxy_box,
                'class_id': 0,
                'class_name': 'person',
                'confidence': score,
                'keypoint': norm_keypoint
            }

            self.pose_data.append(kpt_data)

            # Draw the detection on the input image
            self.draw_detections(input_img, box, score, class_id, keypoint)

        return input_img, self.pose_data


    def predict(self, input_img):

        img_data = self.preprocess(input_img)
        self.inputs[0].host = img_data
        trt_outputs = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)

        return self.postprocess(self.img, trt_outputs)
    


if __name__ == "__main__":

    pose_model = YOLOV8POSE('./yolov8n-pose.trt', 0.5, 0.5)
    out_img, pose_data = pose_model.predict('./bus.jpg')

    cv2.imwrite('output_pose1.jpg', out_img)





        


        