# https://docs.ultralytics.com/usage/cfg/
yolo detect predict\
 model=out/train/n10_yolov8m_i1_run0/weights/best.pt\
 source=data/cow200/test_images\


yolo detect predict\
 model=out/train/n200_yolov8m_i1_run0/weights/best.pt\
 source=data/cow200/test_images\
 imgsz=640\



yolo detect predict\
 model=out/train/n200_yolov8m_i1_run0/weights/best.pt\
 source=data/cow200/yolov5/run0/test.txt




yolo detect predict\
 model=model/yolov8m.pt\
 source=data/cow200/yolov5/run0/test.txt