import cv2
import threading
import time
import csv
import os,glob
from datetime import datetime
from collections import deque
from ultralytics import YOLO
import numpy as np
import natsort
import supervision as sv
from collections import defaultdict
# ------------- 全域狀態 -------------
q2buf_len = 100
frame_queue_cam1 = deque(maxlen=5)
frame_queue_cam2 = deque(maxlen=5)

object_queue_cam1 = deque(maxlen=100)   # 只收 cam1 的「新」ID
object_queue_cam2 = deque(maxlen=q2buf_len)    # 可視需要，保留 cam2 的新ID
matched_queue     = deque(maxlen=100)

output_frame_cam1 = None
output_frame_cam2 = None
matched_crop_cam1 = None
matched_crop_cam2 = None
current_similarity = None

stop_flag_cam1 = False
stop_flag_cam2 = False
lock = threading.Lock()

# 模型與追蹤器
model_path  = "./best_yolov11_PET.pt"#"yolov11n.pt"
tracker_cfg = "botsort.yaml"#"botsort.yaml"
tracker_cfg1 = "bytetrack.yaml"#"botsort.yaml"
# 黑圖（無新檔時補用）
BLACK_IMG_PATH = "/home/gilbert11/Downloads/yolov8-object-tracking/black.jpg"
BLACK_IMG = cv2.imread(BLACK_IMG_PATH) if os.path.exists(BLACK_IMG_PATH) else np.zeros((480, 640, 3), np.uint8)

# CSV 檔（紀錄 cam1<->cam2 對應）
CSV_FILE = "matched_queue10.csv"
CSV_FIELDS = ["timestamp", "cam1_id", "cam1_class", "cam2_id", "cam2_class", "similarity"]
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

cross_line_counts = {
    'cam0': [],
    #'cam2': [],
    'cam4': []
}

cam_combine_flag = 0

# ------------- 工具：形狀相似度（不縮放，直接用原尺寸） -------------
def compare_shape_similarity(img1, img2):
    try:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        _, th1 = cv2.threshold(g1, 55, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, th2 = cv2.threshold(g2, 55, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        c1, _ = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c2, _ = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not c1 or not c2: return 999.0
        cnt1 = max(c1, key=cv2.contourArea)
        cnt2 = max(c2, key=cv2.contourArea)
        return cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I1, 0.0)
    except Exception:
        return 999.0


# ------------- 讀取：影片或資料夾最新檔案（沒新檔就補黑圖） -------------
def read_frames(video_path, frame_queue, stop_flag_name):
    """
    若 video_path 是資料夾：
      - 每 0.1 秒抓最新檔案（依修改時間）
      - 若沒有新檔，不 push 任何 frame（等待下一輪）
    若 video_path 是影片路徑：
      - 正常逐幀讀取，播完結束
    可被外部把 stop_flag_cam1/stop_flag_cam2 設為 True 來結束。
    """
    global stop_flag_cam1, stop_flag_cam2

    # ---------- 資料夾模式 ----------
    if os.path.isdir(video_path):
        import glob
        last_seen_mtime = None

        while True:
            # 允許外部終止
            if (stop_flag_name == "cam1" and stop_flag_cam1) or \
               (stop_flag_name == "cam2" and stop_flag_cam2):
                break

            # 搜索影像檔
            files = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                files.extend(glob.glob(os.path.join(video_path, ext)))
            if files:
                # 依最後修改時間找最新一張
                newest = max(files, key=os.path.getmtime)
                mtime = os.path.getmtime(newest)

                # 有新檔才讀並 append；沒有新檔就什麼都不做（等待）
                if last_seen_mtime is None or mtime > last_seen_mtime:
                    img = cv2.imread(newest)
                    if img is not None:
                        with lock:
                            frame_queue.append(img)
                        last_seen_mtime = mtime
                    else:
                        # 檔案寫入未完成或讀取失敗，稍後再試
                        pass

            time.sleep(0.2)  # 輪詢間隔

        # 結束：標記對應 stop flag
        if stop_flag_name == "cam1":
            stop_flag_cam1 = True
        else:
            stop_flag_cam2 = True
        return

    # ---------- 影片模式 ----------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 無法開啟來源: {video_path}")
        if stop_flag_name == "cam1":
            stop_flag_cam1 = True
        else:
            stop_flag_cam2 = True
        return

    while True:
        # 允許外部終止（例如按下 q）
        if (stop_flag_name == "cam1" and stop_flag_cam1) or \
           (stop_flag_name == "cam2" and stop_flag_cam2):
            break

        ret, frame = cap.read()
        if not ret:
            break
        with lock:
            frame_queue.append(frame)
        time.sleep(0.2)  # 控制讀取節奏

    cap.release()
    if stop_flag_name == "cam1":
        stop_flag_cam1 = True
    else:
        stop_flag_cam2 = True


def save_crop_csv(CAM1_CROP_DIR, CAM1_LOG_CSV, annotated_frame, cam1_object_counter, clss, confe, track_id, x1, x2, y1, y2):
    # ---------------------------------------------------------------
    # 1) 擷取原始 crop（不加字） → 進 queue1 之後會給 cam2 做比對
    crop_raw = annotated_frame[(y1-20):(y2+20), (x1-30):(x2+35)].copy()
    # 2) 產生工作編號
    cam1_object_counter += 1
    obj_no = cam1_object_counter  # 例如 1,2,3...
    filename = f"cam1_obj_{track_id:04d}_0.jpg"
    no_text = f"NO.{obj_no} >tid.{track_id} >{clss} >{filename}"
    # 3) 建立一份要存檔的 crop，右上角壓上 "NO.x"
    crop_to_save = crop_raw.copy()
    if crop_to_save.size != 0:
        
        # 在右上角畫標籤底色(可選)
        h_c, w_c = crop_to_save.shape[:2]
        # 畫一個半透明角落視覺塊也行，不過簡單起見直接文字
        cv2.putText(
            crop_to_save,
            no_text,
            (max(5, w_c - 550), 30),  # 右上角附近
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        # 4) 檔名與寫檔
        
        save_path = os.path.join(CAM1_CROP_DIR, filename)
        cv2.imwrite(save_path, crop_to_save)

        # 5) 寫入 cam1_detect_log.csv
        with open(CAM1_LOG_CSV, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["no", "track_id", "class", "conf", "filename"]
            )
            writer.writerow({
                "no": obj_no,
                "track_id": track_id,
                "class": clss,
                "conf": round(confe, 2),
                "filename": filename
            })
    # ===============================================================
    return cam1_object_counter

def save_crop_csv2(CAM2_CROP_DIR, CAM2_LOG_CSV, annotated_frame, cam2_object_counter, clss, confe, track_id, x1, x2, y1,y2):
    # ---------------------------------------------------------------
    # 1) 擷取原始 crop（不加字） → 進 queue1 之後會給 cam2 做比對
    #crop_raw = annotated_frame[y1:y2, x1:x2].copy()
    crop_raw = annotated_frame[(y1-20):(y2+20), (x1-30):(x2+40)].copy()
    # 2) 產生工作編號
    cam2_object_counter += 1
    obj_no = cam2_object_counter  # 例如 1,2,3...
    filename = f"cam2_obj_{track_id:04d}_0.jpg"
    no_text = f"NO.{obj_no} >tid.{track_id} >{clss} >{filename}"
    #no_text = f"NO.{obj_no}"
    # 3) 建立一份要存檔的 crop，右上角壓上 "NO.x"
    crop_to_save = crop_raw.copy()
    if crop_to_save.size != 0:
        # 在右上角畫標籤底色(可選)
        h_c, w_c = crop_to_save.shape[:2]
        # 畫一個半透明角落視覺塊也行，不過簡單起見直接文字
        cv2.putText(
            crop_to_save,
            no_text,
            (max(5, w_c - 550), 30),  # 右上角附近
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        # 4) 檔名與寫檔
        #filename = f"cam2_obj_{track_id:04d}_0.jpg"
        save_path = os.path.join(CAM2_CROP_DIR, filename)
        cv2.imwrite(save_path, crop_to_save)

        # 5) 寫入 cam2_detect_log.csv
        with open(CAM2_LOG_CSV, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["no", "track_id", "class", "conf", "filename"]
            )
            writer.writerow({
                "no": obj_no,
                "track_id": track_id,
                "class": clss,
                "conf": round(confe, 2),
                "filename": filename
            })
    # ===============================================================
    return cam2_object_counter



def save_combine(CAM_CROP_DIR,  annotated_frame  , cam3_object_counter):
    # ---------------------------------------------------------------
    # 1) 擷取原始 crop（不加字） → 進 queue1 之後會給 cam2 做比對
    #crop_raw = annotated_frame[y1:y2, x1:x2].copy()
    os.makedirs(CAM_CROP_DIR, exist_ok=True)
    crop_to_save = annotated_frame.copy()
    # 2) 產生工作編號
    cam3_object_counter += 1
    obj_no = cam3_object_counter  # 例如 1,2,3...
    filename = f"cam_combine_{cam3_object_counter:04d}.jpg"
    no_text = f"NO.{obj_no} >>{filename}"
    # 3) 建立一份要存檔的 crop，右上角壓上 "NO.x"
    if crop_to_save.size != 0:
        # 4) 檔名與寫檔
        save_path = os.path.join(CAM_CROP_DIR, filename)
        cv2.imwrite(save_path, crop_to_save)
    # ===============================================================
    return cam3_object_counter
    
# ------------- cam1：只在「新 track_id」出現時才入 queue1 -------------
def yolo_worker_cam1():
    global output_frame_cam1
    model = YOLO(model_path)
    seen_ids = set()
    crossed_objects = {}
    # Define the line coordinates
    x11 = 670
    line_coords_list = [(x11, 0, x11, 2480, (0, 255, 255))]
    START = sv.Point(x11, 0)
    END = sv.Point(x11, 2480)
    CAM1_LOG_CSV = "cam131_detect_log.csv"
    CAM1_CROP_DIR = "cam131_crops"
    cam1_object_counter = 0
    os.makedirs(CAM1_CROP_DIR, exist_ok=True)
    track_history = defaultdict(lambda: [])
    while not stop_flag_cam1 or len(frame_queue_cam1) > 0:
        with lock:
            frame = frame_queue_cam1.popleft() if frame_queue_cam1 else None
        if frame is None:
            time.sleep(0.01); continue
        for (x1, y1, x2, y2, color) in line_coords_list:
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        results = model.track(frame, persist=True, tracker=tracker_cfg, verbose=False, conf=0.5)
        #if not results: 
            #with lock: output_frame_cam1 = frame.copy()
            #continue
        # 🔸 使用 BoT-SORT tracking 模式

        result = results[0]
        r = results[0].boxes
        if not hasattr(result, 'boxes'):
             continue
        #print(r)
        cls = r.cls.cpu()
        conf= r.conf.cpu()
        boxes = r.xywh.cpu()
        track_ids = (
                r.id.int().cpu().tolist()
                if r.id is not None
                else None)

        annotated_frame = result.plot()
        
        if track_ids is not None:
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    ######################
                    x1, y1, x2, y2 = int(x-w/2),int(y-h/2),int(x+w/2),int(y+h/2)# map(int, box.xyxy[0])
                    confe = float(r.conf[0])
                    clss = model.names[int(r.cls[0])]
                    ######################
                    print("--------11------------> ",track_id)
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
                    cv2.putText(annotated_frame, f"{track_id,clss}", (x1 - 20, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 255, 255), 3, cv2.LINE_AA)
                    # Check if the object crosses the line
                    crop = annotated_frame[y1:y2, x1:x2].copy()
                    if START.x > ( x1 - ((x2-x1)/2) ):  # Assuming objects cross horizontally
                       if track_id not in crossed_objects and track_id not in cross_line_counts['cam0']  :
                          crossed_objects[track_id] = True
                          cross_line_counts['cam0'].append((track_id, clss,confe))
                          object_queue_cam1.append({
                            "id": track_id,
                            "class": clss,
                            "crop": crop})
                          cam1_object_counter = save_crop_csv(CAM1_CROP_DIR, CAM1_LOG_CSV, annotated_frame,
                                                              cam1_object_counter, clss, confe, track_id, x1, x2, y1,y2)
                          print("----------------------------------------------------")
                          print("====================================================")
                       else:
                          print("=========......................................=====")
                          #object_queue_cam1.append({"id": track_id,"class": clss,"crop": crop}) 
                          # -------------------
                          #cam1_object_counter = save_crop_csv(CAM1_CROP_DIR, CAM1_LOG_CSV, annotated_frame,
                                                              #cam1_object_counter, clss, confe, track_id, x1, x2, y1,y2)
                    time.sleep(0.2)
        '''jj = 0
        for  track_id, clss,confe in   cross_line_counts['cam0']:
             #if track_id not in object_queue_cam1[]
             labels= f"{jj}:>{confe:.2f}:{clss} :ID:>{track_id}"
             cv2.putText(annotated_frame,labels,(x11,50+jj*50),cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,255,255), 2)       
             jj+=1
             if   jj == 50 :
                  jj = 0'''
            #break
        '''for r in results:
            if not hasattr(r, "boxes"): continue
            for b in r.boxes:
                cls = model.names[int(b.cls[0])]
                tid = int(b.id[0]) if b.id is not None else -1
                conf = float(b.conf[0])
                x1,y1,x2,y2 = map(int, b.xyxy[0])

                # 只有新的 track_id 才入 queue1
                if tid not in seen_ids:
                    crop = frame[y1:y2, x1:x2].copy()
                    with lock:
                        object_queue_cam1.append({"id": tid, "class": cls, "crop": crop})
                    seen_ids.add(tid)
                    print(f"🆕 CAM1 新物件: ID={tid}, class={cls}")

                # 繪製
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f"{cls} ID:{tid} {conf:.2f}",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)'''
        with lock: output_frame_cam1 = annotated_frame.copy()#frame.copy()
        time.sleep(0.2)
    idd = 0
    for item in object_queue_cam1:
	    print(idd," : ", len(object_queue_cam1), " :------:>" ,item['id'], item['class'] )
	    idd+=1    
    print("✅ CAM1 結束")



# ------------- cam2：只在「新 track_id」出現時才觸發比對 -------------
def yolo_worker_cam2():
    global output_frame_cam2, matched_crop_cam1, matched_crop_cam2, current_similarity, cam_combine_flag
    model = YOLO(model_path)
    seen_ids = set()
    read_offset = 0
    last_len = 0
    # Define the line coordinates
    x22 = 670
    line_coords_list = [(x22, 0, x22, 2480, (0, 255, 255))]
    CAM2_LOG_CSV = "cam231_detect_log.csv"
    CAM2_CROP_DIR = "cam231_crops"
    cam2_object_counter = 0
    os.makedirs(CAM2_CROP_DIR, exist_ok=True)
    while not stop_flag_cam2 or len(frame_queue_cam2) > 0:
        with lock:
            frame = frame_queue_cam2.popleft() if frame_queue_cam2 else None
        if frame is None:
            time.sleep(0.2); continue
        for (x1, y1, x2, y2, color) in line_coords_list:
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        results = model.track(frame, persist=True, tracker=tracker_cfg1, verbose=False,conf = 0.5)
        if not results:
            with lock: output_frame_cam2 = frame.copy()
            continue

        for r in results:
            #if not hasattr(r, "boxes"): continue
            for b in r.boxes:
                cls = model.names[int(b.cls[0])]
                tid2 = int(b.id[0]) if b.id is not None else -1
                conf = float(b.conf[0])
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                crop2 = frame[y1:y2, x1:x2].copy()

                # 只處理 cam2 的【新】ID
                if tid2 in seen_ids:
                    # 照樣畫框顯示，但不觸發比對
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                    cv2.putText(frame,f"{cls} ID:{tid2} {conf:.2f}",(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)
                    continue
                seen_ids.add(tid2)

                with lock:
                    object_queue_cam2.append({"id": tid2, "class": cls, "crop": crop2})
                    cam1_list = list(object_queue_cam1)
                    n = len(cam1_list) 
                    p = len(object_queue_cam2)                 
                    #print("--------------------cam1_list len <--------------------------------", n)
                    if n == 0 or p == 0 :
                        pass
                    else:
                        #if n != last_len:
                            #read_offset = 0
                            #last_len = n
                        #else:
                        read_offset = min(read_offset + 1, max(0, n - q2buf_len ))
                        #============================================================================
                        read_offset += 1
                        s = read_offset
                        e = min(s + q2buf_len, n)
                        compare_targets = cam1_list[s:e]
                    print("--------------------cam1_list len >--------------------------------", n , s, e)
                    cam2_object_counter = save_crop_csv2(CAM2_CROP_DIR, CAM2_LOG_CSV, frame,
                                                              cam2_object_counter, cls, conf, tid2, x1, x2, y1,y2)
                # 比對
                if n > 0:
                    idx = 0 
                    
                    for ref in compare_targets:
                        sim = compare_shape_similarity(ref["crop"], crop2)
                        print(idx,"--------------------comare >--------",len(compare_targets) , sim ,ref["class"] ,cls )
                        if sim < 0.02 and cls == ref["class"]:
                            # 記錄 matched（保留原尺寸 crop，顯示時不縮放）
                            cam_combine_flag = 1
                            with lock:
                                matched_queue.append({
                                    "cam1_id": ref["id"], "cam1_class": ref["class"],
                                    "cam2_id": tid2,    "cam2_class": cls,
                                    "similarity": sim,
                                    "crop1": ref["crop"], "crop2": crop2
                                })
                                # 就地移除 queue1 中該 cam1 id
                                tid1 = ref["id"]
                                cls1 = ref["class"]
                                filtered = [o for o in object_queue_cam1 if o["id"] != ref["id"]]
                                object_queue_cam1.clear(); object_queue_cam1.extend(filtered)
                                cv2.putText(ref["crop"], f"ID:{tid1}  cls:{cls1}", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                                cv2.putText(crop2, f"ID:{tid2}  cls:{cls} {conf:.2f}", (40, 110), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 2)
                                matched_crop_cam1 = ref["crop"].copy()
                                matched_crop_cam2 = crop2.copy()
                                current_similarity = sim

                            # 寫 CSV：cam1_id, cam1_class, cam2_id, cam2_class, similarity
                            with open(CSV_FILE, "a", newline="") as f:
                                csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "cam1_id": ref["id"], "cam1_class": ref["class"],
                                    "cam2_id": tid2,     "cam2_class": cls,
                                    "similarity": round(sim, 6)
                                })
                            print(f"🔁 MATCH: cam1({ref['id']},{ref['class']}) <-> cam2({tid2},{cls})  sim={sim:.5f}")
                            break
                        cam_combine_flag=0
                # 畫框（cam2）
                #cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                #cv2.putText(frame,f"{cls} ID:{tid2} {conf:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),3)

        with lock: output_frame_cam2 = frame.copy()
        time.sleep(0.2)
    print("✅ CAM2 結束")


# ------------- 拼接：保持原尺寸，不縮放，用黑底畫布擺放 -------------
def make_grid_original(f2, f1, mc2, mc1):
    # 四格：左上 f2、右上 f1、左下 mc2、右下 mc1
    def shape(img):
        return img.shape[:2] if img is not None else (0, 0)
    h_tl, w_tl = shape(f2)
    h_tr, w_tr = shape(f1)
    h_bl, w_bl = shape(mc2)
    h_br, w_br = shape(mc1)

    col_w = [max(w_tl, w_bl), max(w_tr, w_br)]
    row_h = [max(h_tl, h_tr), max(h_bl, h_br)]
    H = row_h[0] + row_h[1]
    W = col_w[0] + col_w[1]

    canvas = np.zeros((max(H,1), max(W,1), 3), dtype=np.uint8)

    def paste(img, x, y):
        if img is None: return
        h, w = img.shape[:2]
        canvas[y:y+h, x:x+w] = img

    # 左上、右上、左下、右下
    paste(f2, 0, 0)
    paste(f1, col_w[0], 0)
    paste(mc2, 0, row_h[0])
    paste(mc1, col_w[0], row_h[0])

    return canvas


# ------------- 顯示：保持原尺寸 + WINDOW_KEEPRATIO（不 resize） -------------
def display_combined():
    global cam_combine_flag
    cv2.namedWindow("Multi-Camera Match Display", cv2.WINDOW_KEEPRATIO)
    output_path = "2025-1024_pet_result_v1-131.mp4"  # Output video file path
    output_dir = "2025-1024_pet_result_v1-131"
    CAM_CROP_DIR = "cam131_combine"
    cam3_object_counter = 0
    #frame_width = 2448
    #frame_height = 2048
    os.makedirs(output_dir, exist_ok=True)
    #out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 1, (frame_width, frame_height))
    imgdir= []
    idx = 0
    while True:
        with lock:
            f1 = output_frame_cam1.copy() if output_frame_cam1 is not None else None
            f2 = output_frame_cam2.copy() if output_frame_cam2 is not None else None
            mc1 = matched_crop_cam1.copy() if matched_crop_cam1 is not None else None
            mc2 = matched_crop_cam2.copy() if matched_crop_cam2 is not None else None
            sim = current_similarity

            # 於 cam2 左上角列 queue 狀態（ID, class）
            if f2 is not None:
                base_y = f2.shape[0] - 180
                q1 = [(o["id"], o["class"]) for o in list(object_queue_cam1)]
                q2 = [(o["id"], o["class"]) for o in list(object_queue_cam2)]
                cv2.putText(f2, "CAM2 (BoT-SORT)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
                cv2.putText(f2, f"Queue1({len(q1)}):", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                for i,(tid,cl) in enumerate(q1[-50:]):
                    cv2.putText(f2, f"ID={tid}, {cl}", (60, 150+i*38),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
                cv2.putText(f2, f"Queue2({len(q2)}):", (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                for i,(tid,cl) in enumerate(q2[-50:]):
                    cv2.putText(f2, f"ID={tid}, {cl}", (400, 150+i*38),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)

            if f1 is not None:
                cv2.putText(f1, "CAM1 (BoT-SORT)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
                q1 = [(o["id"], o["class"]) for o in list(object_queue_cam1)] 
                cv2.putText(f1, f"Queue1({len(q1)}):", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                for i,(tid,cl) in enumerate(q1[-50:]):
                    cv2.putText(f1, f"ID={tid}, {cl}", (60, 150+i*38),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)

            if mc1 is not None and sim is not None:
                cv2.putText(mc1, f"MATCH CROP CAM1 > sim:{sim:.5f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            if mc2 is not None:
                cv2.putText(mc2, f"MATCH CROP CAM2 > sim:{sim:.5f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        
        combined = make_grid_original(f2, f1, mc2, mc1)
        if mc1 is not None and  mc2 is not None and sim is not None and cam_combine_flag == 1:
           cam3_object_counter = save_combine(CAM_CROP_DIR,  combined  , cam3_object_counter)
           cam_combine_flag =0
        #frame_height, frame_width= combined.shape[:2]
        cv2.imshow("Multi-Camera Match Display", combined)
        output_filename = os.path.join(output_dir, str(idx)+ ".jpg" )
        imgdir.append(output_filename)
        idx+=1
        cv2.imwrite(output_filename,combined)
        #print(frame_height,frame_width,"<h:w>*************************************************************************")
        #out.write( cv2.imread(os.path.join(output_dir, str(idx)+ ".jpg")) )
        key = cv2.waitKey(1000) & 0xFF
        if key == ord('q'):
            # 結束所有 thread
            global stop_flag_cam1, stop_flag_cam2
            stop_flag_cam1 = True
            stop_flag_cam2 = True
            break
        time.sleep(0.2)
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (4096, 2448))#w,h  
    for im in imgdir:  
        out.write( cv2.imread(im) )    
    out.release()
    cv2.destroyAllWindows()
    print("🖥️ 顯示結束")


# ------------- 主程式 -------------
def main(source_cam1, source_cam2):
    t_r1 = threading.Thread(target=read_frames, args=(source_cam1, frame_queue_cam1, "cam1"))
    t_r2 = threading.Thread(target=read_frames, args=(source_cam2, frame_queue_cam2, "cam2"))
    t_y1 = threading.Thread(target=yolo_worker_cam1)
    t_y2 = threading.Thread(target=yolo_worker_cam2)
    t_ui = threading.Thread(target=display_combined)

    for t in (t_r1, t_r2, t_y1, t_y2, t_ui): t.start()
    for t in (t_r1, t_r2, t_y1, t_y2, t_ui): t.join()

    print("🎯 全流程完成")


if __name__ == "__main__":
    # source_cam1 / source_cam2 可是「影片路徑」或「資料夾路徑」
    # 例：source_cam1 = "camera1_frames/"（資料夾），source_cam2 = "camera2.mp4"（影片）
    source_cam1 = "/home/gilbert11/Downloads/yolov8-object-tracking/2.mp4"#"/home/gilbert11/Downloads/yolov8-object-tracking/tracking0918/45"
    source_cam2 = "/home/gilbert11/Downloads/yolov8-object-tracking/3.mp4"#"/home/gilbert11/Downloads/yolov8-object-tracking/tracking0918/67"
    main(source_cam1, source_cam2)

