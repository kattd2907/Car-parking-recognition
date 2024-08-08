from ultralytics import YOLO
import csv

model = YOLO(r'C:\Users\Admin\PycharmProjects\pythonProjectest\yolo\best.pt')

results = model.predict(source=r'C:\Users\Admin\PycharmProjects\pythonProjectest\images\3.jpg', imgsz=640, save=True)


output_csv = r'C:\Users\Admin\PycharmProjects\pythonProjectest\csv\detect.csv'
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["label", "confidence", "xmin", "ymin", "xmax", "ymax"])

    for result in results:
        boxes = result.boxes  # Bounding boxes
        labels = result.names  # Class labels
        for box in boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0]
            label = labels[int(box.cls)]
            confidence = box.conf
            writer.writerow([label, confidence, xmin, ymin, xmax, ymax])

