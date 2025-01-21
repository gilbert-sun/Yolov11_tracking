'''
import cv2
from ultralytics import YOLO

model = YOLO('/home/gilbert11/Documents/mimii_evaluation_GUI/runs/detect/train8/weights/best.pt')  # Load an official Detect model
 
# Perform tracking with the model
results = model.predict(source="flat_pet.avi", show=True, save=False, imgsz=640, stream=True, classes=7)  # Tracking with default tracker
'''

from collections import defaultdict
import cv2,os
import numpy as np
from ultralytics import YOLO

def track_video(video_path):
    # load the model
    os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(160000)
    model = YOLO("/home/gilbert11/Documents/mimii_evaluation_GUI/runs/detect/train8/weights/best.pt")#"/home/gilbert11/Downloads/ultralytics/yolov8n.pt")#
    # open the video file
    cap = cv2.VideoCapture(video_path)#"./all_pet.avi")
    track_history = defaultdict(lambda: [])
    # get the video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width =  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    # define the codec and create VideoWriter object
    output_path = "all_pet_v8.mp4"  # Output video file path
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps*2, (frame_width, frame_height)
    )
    # loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True)
            boxes = results[0].boxes.xywh.cpu()
            track_ids = (
                results[0].boxes.id.int().cpu().tolist()
                if results[0].boxes.id is not None
                else None
            )
            annotated_frame = results[0].plot()
            # plot the tracks
            if track_ids:
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)
                    # draw the tracking lines
                    points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(230, 230, 230),
                        thickness=2,
                    )
            # write the annotated frame   
            cv2.imshow("PET_FLAT",annotated_frame)#frame
            out.write(annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    # release the video capture object and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path
if __name__ == "__main__":
    
	print("Entering ... ... ")    
	track_video("/home/gilbert11/Downloads/yolov8-object-tracking/flat_pet.avi")#GH012601.MP4")#flat_pet.avi")#./GH010203.MP4")#GH012601.MP4")#./2021-02-03 14-29-00.mp4")#"./flat_pet.avi")    
	print("Exit ...")
