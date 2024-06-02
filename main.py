from model.MultiModel import MultiTaskModel
import yaml
import torch
import time
from Track.Tracker import Detection, Tracker
from Pose import YOLOV8POSE
from Souces import CamLoader, CamLoader_Q
import argparse
import cv2
import os
import numpy as np

LABELS_NAME =  ['Normal', 'Fall']


def make_outdir(out_path="./data/predit/output"):
    """
        保存每次运行结果
        创建输出文件夹，如果文件夹存在，则新建 "输出文件夹名称_n"
    """
    output_dir = out_path

    if os.path.exists(output_dir):
        i = 1
        while True:
            new_output_dir = f"{out_path}_{i}"
            if not os.path.exists(new_output_dir):
                output_dir = new_output_dir
                break
            i += 1

    os.makedirs(output_dir)

    return output_dir


def find_video_file(directory):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']
    video_files = []

    for file in os.listdir(directory):
        if file.lower().endswith(tuple(video_extensions)):
            video_files.append(os.path.join(directory, file))
    
    return video_files


def main(args, video, out_path):
    # init model   
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda')
    action_model = MultiTaskModel(config['model_args'], config['head_args']).to(device)
    print(f'loading model...')
    action_model.load_state_dict(torch.load('./multi_b64l6d35i35.pth', map_location=device))
    action_model.eval()

    tracker = Tracker()
    pose_model = YOLOV8POSE('./yolov8n-pose.trt')

    # load video
    # out_path = make_outdir()
    out_video = args.save_out

    # Read data from video(path) or camera(rtsp, index:0, 1, ...)
    # Input: video(path) or camera(rtsp, index:0, 1, ...) : str
    # Output: video stream.
    cam_source = video
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source).start()

    if out_video:
        out_name = video.split('/')[-1].split('.')[0]
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(out_path + f'/output_{out_name}.mp4', codec, cam.fps, cam.frame_size)
        print(f'CAM FPS: {cam.fps}')



    frame_index = 1
    while cam.grabbed():
        start = time.time()
        frame = cam.getitem()
        image = frame.copy()

        pose_start = time.time()

        out_img, pose_data = pose_model.predict(frame)
        # print(len(pose_results))
        pose_results = sorted(pose_data, key=lambda x : x['confidence'], reverse=True)
        # only select few person 
        pose_results = pose_results[:args.max_person]

        pose_end = time.time()
        print(f'pose cost : {pose_end - pose_start}')
        tracker.predict()
        # detected_np = np.array([track.to_tlbr().tolist() + [0.5, 1.0, 0.0] for track in tracker.tracks], dtype=np.float32)
        # detected = np.concatenate([detected, detected_np], axis=0) if detected is not None else detected_np

        detections = [Detection(np.array(pose["box"]), 
                                np.array(pose["keypoint"]), 
                                np.array(pose["confidence"])) 
                      for pose in pose_results]

        tracker.update(detections)
        # print(f'yolov8 out : {pose_results}')

        # predit every track
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)
            print(f'track length : {len(track.keypoints_list)}')

            action = 'Loading...'
            clr = (0, 255, 0)
            # action predit
            if len(track.keypoints_list) >= 40:
                pts = np.array(track.keypoints_list, dtype=np.float32)[:,:, :2].reshape(1,1,-1,17,2)
                kpt_input = torch.from_numpy(pts).to(device)
                print(kpt_input.shape)
                t1 = time.time()
                out = action_model.cls(kpt_input)
                t2 = time.time()
                print(f'action predit cost: {t2-t1}')
                print(f'out shape: {out.shape} and out type: {type(out)} \nout: {out}')

                action_name = LABELS_NAME[out.argmax()]
                # action = f'{track.track_id}_{action_name}'
                action = action_name
                if action_name == 'Fall':
                    clr = (0, 0, 255)
                elif action_name == 'LyingDown':
                    clr = (0, 200, 255)
                
                cv2.putText(out_img, action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, clr, 2)
                cv2.putText(out_img, f'Action_FPS: {1/(t2-t1):.1f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            print(f'{frame_index} : {action}')

        end = time.time()
        cost_time = (end - start)
        fps = 1 / cost_time
        print(f'{cost_time}, {fps}')
        cv2.putText(out_img, f'Pose_FPS: {1/(pose_end-pose_start):.1f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(out_img, f'{frame_index}___FPS:{fps:.1f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if out_video:
            writer.write(out_img)
        # cv2.imwrite(f"{out_path}/{frame_index}.jpg", out_img)
        frame_index += 1
        print(f'one frame cost: {(end - start) * 1000:.2f}ms')


    cam.stop()
    if out_video:
        writer.release()

if __name__ == "__main__":

    # init arg parser
    parse = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    # rtsp :rtsp://admin:Admin123456@192.168.1.50:554/stream1
    parse.add_argument('-C', '--camera', default="/home/hnu2021/tmp/yolov8/fall.avi", 
                        help='Source of camera or video file path.')
    parse.add_argument('--detection_input_size', type=int, default=640,
                        help='Size of input in detection model in square must be divisible by 32 (int).')
    parse.add_argument('--save_out', type=bool, default=True,
                    help='Save display to video file.')
    parse.add_argument('--max_person', type=int, default=1,
                    help='Maximum detection number per frame')
    args = parse.parse_args()

    out_path = make_outdir()

    if os.path.isdir(args.camera):
        file_list = find_video_file(args.camera)
        print(len(file_list))
        print(file_list)

        for file in file_list:
            main(args, file, out_path)
    else:
        main(args, args.camera, out_path)


   