import cv2
from ultralytics import YOLO

def POINTS(event, x, y, flags, param):
    global roi_points
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        if len(roi_points) == 2:
            cv2.rectangle(frame, roi_points[0], roi_points[1], (0, 255, 0), 2)
            cv2.putText(frame, "ROI", (roi_points[0][0], roi_points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
            cv2.imshow("ROI", frame)
            cv2.waitKey(0)


cv2.namedWindow('ROI' , cv2.WINDOW_NORMAL)
cv2.setMouseCallback('ROI', POINTS)

image_path = r'C:\Users\Admin\PycharmProjects\pythonProjectest\images\1.jpg'

frame = cv2.imread(image_path)

roi_points = []

cv2.imshow("ROI", frame)
cv2.waitKey(0)

x1, y1 = roi_points[0]
x2, y2 = roi_points[1]

roi = frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

model = YOLO(r'C:\Users\Admin\PycharmProjects\pythonProjectest\yolo\yolov8m.pt')

results = model(roi)

num_boxes = 0

for result in results:
    boxes = result.boxes
    num_boxes += len(boxes)
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        label = result.names[int(box.cls)]
        confidence = float(box.conf)

        cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(roi, f"So luong xe do sai: {num_boxes}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

cv2.imshow("ROI", roi)
print(f"So luong xe do sai: {num_boxes}")
cv2.waitKey(0)

cv2.destroyAllWindows()
