## Create Virtual Environment and requirement.txt
sudo apt-get install python3.9-venv

python -m venv windturbine

source windturbine/bin/activate

pip3 freeze > requirements.txt

## labelimg / label format
'class_id x y w h'

## class_id
'P,O,S,C,Ot,T,Ch'

## split data 
python3 cute_data.py #train == 0.9 / val== 0.0/ test == 0.1

## Activate your environment
source Documents/mimii_evaluation_GUI/mimii_gui/bin/activate

## Install requirement library
pip3 install ultralytics

## Training step for v7
python3 train.py --weights ./runs/train/exp2/weights/last.pt  --cfg cfg/training/yolov7.yaml --data data/smoke.yaml --device 0 --batch-size 8 --epoch 50

## Training step for v5/v8/v11
yolo detect train data=pet_dataset_v1.yaml model=yolov8n.pt epochs=50 imgsz=640 device=0
yolo detect train data=pet_dataset_v1.yaml model=yolov11n.pt epochs=100 imgsz=640 device=0
```
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(data='path yo your data.yaml', 
                      epochs=100,
                      batch=64, 
                      imgsz=640, 
                      device=0, 
                      )
```


## Testing step
python3 test.py --weights ./runs/train/exp2/weights/best.pt --task test --data data/smoke.yaml


## Testing step for v5/v8/v11
python3 test.py --weights ./runs/train/exp2/weights/best.pt --task test --data data/smoke.yaml

yolo detect predict model=/home/gilbert11/Documents/mimii_evaluation_GUI/runs/detect/train8/weights/best.pt source=/home/gilbert11/Downloads/ultralytics/ultralytics/cfg/pet_dataset_v1/images/test save=true

(mimii_gui) root@gilbert11-MS-7972:/home/gilbert11/Downloads/ultralytics/ultralytics# yolo detect train data=pet_dataset_v3.yaml model=/home/gilbert11/Documents/mimii_evaluation_GUI/runs/detect/train25/weights/last.pt epochs=25 imgsz=640 device=0 batch=3

## Tracking your Pet_flat
python3 yolov8tracking.py

## Transfrom img 2 video
python3 img2video.py

```
from ultralytics import YOLO

model = YOLO("best.pt")

source="test/images"

result = model.predict(source, imgsz=640, stream=True, save=True)
```

