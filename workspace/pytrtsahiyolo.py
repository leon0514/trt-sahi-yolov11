import trtsahiyolo
from trtsahiyolo import YoloType as YoloType
import cv2

cap = cv2.VideoCapture('test.mp4')

instance = trtsahiyolo.TrtSahiYolo("phone.engine", YoloType.YOLOV5, 0)


index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        result = instance.autoSliceForward(frame)
        if len(result) != 0:
            print(result)
            for res in result:
                left = res.left
                top = res.top
                right = res.right
                bottom = res.bottom
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
            cv2.imwrite(f"auto/{index}.jpg", frame)
            index += 1
    else:
        break
