## Train
0. modify ultrasonic/cfg/dataset/tsd_dataset_v5.yaml to dataset

0.5 sic_data_2yolo_2coco.py cp oringinal folder file (gray,koh,gt) to another folder & transform lefe up cordinator to yolo format cordinator 

1. yolo detect train data=tsd_dataset_v5.yaml model=/home/k900/Documents/ultralytics/runs/detect/train7/weights/last.pt epochs=200 imgsz=640 device=0
===> runs/detect/train1/weights/best.pt


## Testing step1
2. yolo_rotate_aug.py ===> aug 6 times
3. cut_data.py        ===> train/validate/test 8:1:1


## Testing step2
4. yolo detect predict model=/home/k900/Documents/ultralytics/runs/detect/train6/weights/best.pt source=/media/k900/PlextorSSD0/111925_new_forprediction/output_cls0  conf=0.5 save=true save_txt=true save_crop=true
===> runs/detect/predict1/labels , runs/detect/predict1/crops

yolo detect predict model=/home/k900/Documents/ultralytics/runs/detect/train6/weights/best.pt source=/media/k900/PlextorSSD0/111925_new_forprediction_elimit_non0  conf=0.001 save=true save_txt=true save_crop=true


## Testing step3
45.pick0class.py      ===> pick up all 0 class
5. cp_file2folder.py  ===> cp gray & koh picture from test_a to test_b dataset
6. plot_gt1.py        ===> plot all  gt bounding box to gt.png, gray.png, koh.png and predict.png
7. test_f1_score.py   ===> get all f1,acc,recall,predict score
8. preview_yolov11.py ===> see 4 picture koh,gray,gt,predict in one app preview page
9. show_diff_folder_file.py ===> show 2 folder diff files
10../batch_eval_epochs_conf.sh
11.python3 collect_all_results.py
12.python3 recalc_plot_paper5.py --legend_inside --sort metric --no_long_legend --topk_table 8

