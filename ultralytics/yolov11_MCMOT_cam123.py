# -*- coding: utf-8 -*-
import cv2,time
import numpy as np
import threading
from ultralytics import YOLO
from collections import defaultdict
from queue import Queue
from collections import deque


pause_flag = threading.Event()
stop_flag = threading.Event()

latest_outputs = {
    'cam0': [],
    'cam2': [],
    'cam4': []
}

cross_line_counts = {
    'cam0': [],
    'cam2': [],
    'cam4': []
}

CROSS_LINE_X = {
    'cam0': 70,
    'cam2': 570,
    'cam4': 600
}
   


def get_color(idx):
    np.random.seed(idx)
    return tuple(np.random.randint(0, 255, 3).tolist())


def merge_and_output_sync(q0, q2, q4, output_path,fps): 
    out = None
    while True:
        if stop_flag.is_set():
            break

        f0 = q0.get()
        f2 = q2.get()
        f4 = q4.get()

        if f0[1] is None or f2[1] is None or f4[1] is None:
            break

        f0 = f0[1]
        f2 = f2[1]
        f4 = f4[1]

        h = min(f0.shape[0], f2.shape[0], f4.shape[0])
        w = min(f0.shape[1], f2.shape[1], f4.shape[1])
        f0 = cv2.resize(f0, (w, h))
        f2 = cv2.resize(f2, (w, h))
        f4 = cv2.resize(f4, (w, h))

        combined = np.hstack((f0, f2, f4)) # R-M-L 排列

        if out is None:
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                  (combined.shape[1], combined.shape[0]))

        out.write(combined)
        cv2.imshow("Multi-Camera View", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            pause_flag.set()
        elif key == ord('q'):
            stop_flag.set()
            break
        elif key == ord('r'):
            pause_flag.clear()

    if out:
        out.release()
    cv2.destroyAllWindows()
    
def judge1(results,model,current_ids,trajectories,frame,ret,label,passed_ids,output_ids_classes):
            #result= results[0]
            #value = results[0].boxes.id.int().cpu().tolist()
            #print("\n=====================================================================================>value >> " , type (value),value)
            #track_id = int(latest_outputs.get('cam0', [])[-1:][0][0])
            #for box,   cls in zip(results[0].boxes.xywh.int().cpu().tolist(),  result.boxes.cls):
            id_count = 0
            for box,  track_id , cls in zip(results[0].boxes.xywh.int().cpu().tolist(), results[0].boxes.id.int().cpu().tolist() ,  results[0].boxes.cls):
                result= results[0]
                value = results[0].boxes.id.int().cpu().tolist()
                print("\n=====================================================================================>value >> " , type (value),value)
            #results[0].boxes.id.int().cpu().tolist() , 
            #track_id ,
                cx, cy, w, h = box #map(int, box)
                #track_id = int(track_id)
                if id_count == 0:
                   track_id = value[0]
                elif id_count == 1 :
                   track_id = value[1]
                elif id_count == 2 :
                   track_id = value[2]
                else:
                   track_id = value[3]                   
                id_count += 1
                priousID = track_id
                #print(id_class_input, "======----------------> ", priousID)
                class_id = int(cls)
                class_name = model.names[class_id]
                #cx = (x1 + x2) // 2
                #cy = (y1 + y2) // 2
                current_ids.add(track_id)
                trajectories[track_id].append((int(cx), int(cy)))

                color = get_color(track_id)
                cv2.rectangle(frame, (int(cx-cx//2), int(cy-cy//2)), (int(cx+cx//2), int(cy+cy//2)), color, 2)
                cv2.putText(frame, f'{class_name} ID {track_id}', (int(cx-cx//2), int(cy-cy//2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 判斷是否越過跨線
                if label in CROSS_LINE_X and track_id not in passed_ids:
                    if label in ['cam0'] and cx > CROSS_LINE_X[label]:
                        passed_ids.add(track_id)
                        cross_line_counts[label].append((track_id, class_name))
                output_ids_classes.append((track_id, class_name))



def judge2(results,model,current_ids,trajectories,frame,ret,label,passed_ids,output_ids_classes):
            result= results[0]
            #track_id = int(latest_outputs.get('cam0', [])[-1:][0][0])
            print("1======----------------> ",latest_outputs.get('cam0', [])[:])
            print("7======c2--------------->",latest_outputs.get('cam2', [])[:])
            print("\n")
            print("2======----------------> ",latest_outputs.get('cam0', [])[-1:])
            print("3======----------------> ",latest_outputs.get('cam0', [])[-1:][0])
            print("4======----------------> ",latest_outputs.get('cam0', [])[-1:][0][0])
 
            #for box,  track_id , cls in zip(results[0].boxes.xywh.int().cpu().tolist(), results[0].boxes.id.int().cpu().tolist() ,  result.boxes.cls):
            id_count2 = 0             
            for box, cls in zip(results[0].boxes.xywh.int().cpu().tolist(),result.boxes.cls): 
            #track_id ,
            #results[0].boxes.id.int().cpu().tolist()  , 
                cx, cy, w, h = box #map(int, box)
                if  len(latest_outputs.get('cam2', [])) == 0 :
                    track_id = latest_outputs.get('cam0', [])[-1:][0][0]
                    print("---------------------------------------> ", id_count2 , track_id)
                elif id_count2 == 1 and len(latest_outputs.get('cam0', [])) >= 2 :
                    track_id = latest_outputs.get('cam2', [])[-2:-1][0][0]
                else:
                    if cx > 550:
                       track_id = latest_outputs.get('cam0', [])[-1:][0][0]
                    else:
                       track_id = latest_outputs.get('cam0', [])[-2:-1][0][0]   
                id_count2 += 1
                '''
                c1 = latest_outputs.get('cam0', [])[-1:][0][0]
                c10 = latest_outputs.get('cam2', [])[:]
                c20 = latest_outputs.get('cam2', [])[:]
                if c20 != [] :
                  c2 = latest_outputs.get('cam2', [])[-1:][0][0]
                  if len(c10) > len(c20):
                     track_id = latest_outputs.get('cam0', [])[len(c10)-1:len(c10)][0][0]
                  elif (len(c10) == len(c20)):
                     track_id = latest_outputs.get('cam0', [])[-1:][0][0]'''
                #track_id = int(track_id)
                priousID = track_id
                #print(id_class_input, "======xxxxxxxx----------------> ", priousID,type(track_id),track_id)
                class_id = int(cls)
                class_name = model.names[class_id]
                #cx = (x1 + x2) // 2
                #cy = (y1 + y2) // 2
                current_ids.add(track_id)
                trajectories[track_id].append((int(cx), int(cy)))

                color = get_color(track_id)
                cv2.rectangle(frame, (int(cx-cx//2), int(cy-cy//2)), (int(cx+cx//2), int(cy+cy//2)), color, 2)
                cv2.putText(frame, f'{class_name} ID {track_id}', (int(cx-cx//2), int(cy-cy//2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 判斷是否越過跨線
                if label in CROSS_LINE_X and track_id not in passed_ids:
                    if label in ['cam2'] and cx > CROSS_LINE_X[label]:
                        passed_ids.add(track_id)
                        cross_line_counts[label].append((track_id, class_name))
                    elif label == 'cam2' and cx < 270:#CROSS_LINE_X[label]:
                        passed_ids.add(track_id)
                        cross_line_counts[label].append((track_id, class_name))

                output_ids_classes.append((track_id, class_name))


def judge3(results,model,current_ids,trajectories,frame,ret,label,passed_ids,output_ids_classes):
            result= results[0]
            #track_id = int(latest_outputs.get('cam2', [])[-1:][0][0])
            print("11======----------------> ",latest_outputs.get('cam2', [])[:])
            print("17======c3--------------->",latest_outputs.get('cam4', [])[:])
            print("\n")
            print("12======----------------> ",latest_outputs.get('cam2', [])[-1:])
            print("13======----------------> ",latest_outputs.get('cam2', [])[-1:][0])
            print("14======----------------> ",latest_outputs.get('cam2', [])[-1:][0][0])

            #for box,  track_id , cls in zip(results[0].boxes.xywh.int().cpu().tolist(), results[0].boxes.id.int().cpu().tolist() ,  result.boxes.cls):
            id_count3 = 0   
            for box, cls in zip(results[0].boxes.xywh.int().cpu().tolist(),result.boxes.cls): 
            #track_id ,
            #results[0].boxes.id.int().cpu().tolist()  
                #track_id = latest_outputs.get('cam2', [])[-1:][0][0]
                cx, cy, w, h = box #map(int, box)
                if  len(latest_outputs.get('cam4', [])) == 0 :
                    track_id = latest_outputs.get('cam2', [])[0:1][0][0]
                    print("---------------------------------------> ", id_count3, track_id)
                elif id_count3 == 1 and len(latest_outputs.get('cam2', [])) >= 2 :
                    track_id = latest_outputs.get('cam4', [])[-2:-1][0][0]
                else:
                    if cx > 600:
                       track_id = latest_outputs.get('cam2', [])[-1:][0][0]
                    else:
                       track_id = latest_outputs.get('cam2', [])[-2:-1][0][0]   
                id_count3 += 1                
                #track_id = val3                        
                priousID = track_id
                #print(id_class_input, "======----------------> ", priousID)
                class_id = int(cls)
                class_name = model.names[class_id]
                #cx = (x1 + x2) // 2
                #cy = (y1 + y2) // 2
                current_ids.add(track_id)
                trajectories[track_id].append((int(cx), int(cy)))

                color = get_color(track_id)
                cv2.rectangle(frame, (int(cx-cx//2), int(cy-cy//2)), (int(cx+cx//2), int(cy+cy//2)), color, 2)
                cv2.putText(frame, f'{class_name} ID {track_id}', (int(cx-cx//2), int(cy-cy//2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 判斷是否越過跨線
                if label in CROSS_LINE_X and track_id not in passed_ids:
                   if label == 'cam4' and cx < CROSS_LINE_X[label]:
                        passed_ids.add(track_id)
                        cross_line_counts[label].append((track_id, class_name))

                output_ids_classes.append((track_id, class_name))


def track_video_sync(video_path, model_path, line_coords_list, queue_out, label, sync_barrier, id_class_input=None, output_order='LMR'):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    tracker_config = {'tracker': 'botsort.yaml', 'persist': True}
    trajectories = defaultdict(list)
    active_ids = set()
    passed_ids = set()
    current_ids = set()
    output_ids_classes = [] 
    frame_id = 0
    priousID = 0
    cam2_id = 0
    cam3_id = 0
    track_id = 0 
    while not stop_flag.is_set():
        sync_barrier.wait()
        while pause_flag.is_set():
            if stop_flag.is_set():
                break
        if stop_flag.is_set():
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if label == 'cam0':
          results = model.track(frame, persist=True,  conf=0.625)
          #results = model.track( verbose=False, conf=0.3, **tracker_config)
        elif label == 'cam2':
          results = model.track(frame, persist=True,  conf=0.64)
        elif label == 'cam4':
          results = model.track(frame, persist=True,  conf=0.71)                
        result = results[0]


        if result.boxes.id is not None  and id_class_input == 'cam0' :
            judge1(results,model,current_ids,trajectories,frame,ret,label,passed_ids,output_ids_classes)

                
        elif result.boxes.id is not None  and id_class_input == 'cam2' :
            judge2(results,model,current_ids,trajectories,frame,ret,label,passed_ids,output_ids_classes)
         
                
        elif result.boxes.id is not None  and id_class_input == 'cam4' :
            judge3(results,model,current_ids,trajectories,frame,ret,label,passed_ids,output_ids_classes)
              
        #if len(trajectories)> 30:
            #trajectories.pop(track_id,0)

        for track_id in active_ids - current_ids:
            print("track_id-------> ", track_id)
            trajectories.pop(track_id, None)
            
        active_ids = current_ids

        for (x1, y1, x2, y2, color) in line_coords_list:
            cv2.line(frame, (x1, y1), (x2, y2), color, 1)

        cv2.putText(frame, f'{label}_Frame {frame_id}', (frame.shape[1] - 250, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if label == 'cam0':
            latest_outputs['cam0'] = cross_line_counts['cam0']
        elif label == 'cam2':
            latest_outputs['cam2'] = cross_line_counts['cam2']
        elif label == 'cam4':
            latest_outputs['cam4'] = cross_line_counts['cam4']

        if id_class_input:
            _input = latest_outputs.get(id_class_input, [])
            ii = 0
            for tid, cname in _input:
                cv2.putText(frame, f"{label} : {cname} ID {tid}", (10, 20+ ii * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                ii+=1

        jj =0
        label1 = ""
        if label == "cam0":
           label1 = "cam0"
        elif label == "cam2":
           label1 = "cam0"
        elif label == "cam4":
           label1 = "cam2"
        for i, (tid, cname) in enumerate(cross_line_counts[label1]):
            cv2.putText(frame, f"{label1} : {cname} ID {tid}", (220, 20 + jj * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            jj+=1
        queue_out.put((label, frame))
        
        #print("lb, framequeue_out ---------------------", label, " == " ,frame , " -- " ,queue_out[-1:] )
        if (frame_id) > 145:     
            time.sleep(0.01)
        if (frame_id) > 700:
            cap.release()
            cv2.destroyAllWindows()
            break

    queue_out.put((label, None))
    cap.release()
    cv2.destroyAllWindows()


 
    
if __name__ == "__main__":
        #parser = argparse.ArgumentParser()
        #parser.add_argument('--video_folder', type=str, default='/home/k900/Downloads/60885.0_20250410_161050')
        #parser.add_argument('--result_folder', type=str, default="out_padded_img")
        #parser.add_argument('--width', type=int, default=8320)
        #parser.add_argument('--channel', type=int, default=3)
        #parser.add_argument('--data_type', type=str, default='uint8')
        #args = parser.parse_args()
	# 組態與執行緒啟動
	fps_out = 30
	fs_out = 'output_sync_all_cams_0620_fps{}.mp4'.format(fps_out)
	sync_barrier = threading.Barrier(3)
	queue_0, queue_2, queue_4 = Queue(), Queue(), Queue()

	threads = [
	    threading.Thread(target=track_video_sync, args=(
		'./camera_4.avi', './best_yolov11_PET.pt',
		[(70, 0, 70, 480, (0, 255, 255))],
		queue_4, 'cam0', sync_barrier, 'cam0')),


	    threading.Thread(target=track_video_sync, args=(
		'./camera_2.avi', './best_yolov11_PET.pt',
		[(200, 0, 200, 480, (0, 255, 0)), (570, 0, 570, 480, (0, 255, 255))],
		queue_2, 'cam2', sync_barrier, 'cam2')),

	    threading.Thread(target=track_video_sync, args=(
		'./camera_0.avi', './best_yolov11_PET.pt',
		[(500, 0, 500, 480, (0, 255, 0))],
		queue_0, 'cam4', sync_barrier, 'cam4'))
	]

	merge_thread = threading.Thread(target=merge_and_output_sync,
		                         args=(queue_0, queue_2, queue_4, fs_out ,fps_out))

	for t in threads:
	    t.start()
	merge_thread.start()

	for t in threads:
	    t.join()
	merge_thread.join()

	
