from ultralytics import YOLO
import cv2

model = YOLO(r'C:\Users\Admin\PycharmProjects\pythonProjectest\yolo\best.pt')
img = cv2.imread(r'C:\Users\Admin\PycharmProjects\pythonProjectest\images\1.jpg')
results = model.predict(img, stream=True)

counter = 0
for result in results:
    boxes = result.boxes.cpu().numpy()
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        if class_name == "car":
            r = box.xyxy[0].astype(int)
            crop = img[r[1]:r[3], r[0]:r[2]]
            cv2.imwrite(f'image_crop/car_{counter}.jpg', crop)
            counter += 1
