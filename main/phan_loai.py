from ultralytics import YOLO


model = YOLO(r'C:\Users\Admin\PycharmProjects\pythonProjectest\yolo\yolov8m-seg.pt')

results = model.predict(source=r'C:\Users\Admin\PycharmProjects\pythonProjectest\image_crop\car_11.jpg', imgsz=640, save=True)
pixels_ps = 285

boxes = results[0].boxes.xywh.cpu()
for box in boxes:
    x, y, w, h = box[:4]
    print(" Chieu dai Box: {}".format(h))
    print("Ti le: {}".format(h/pixels_ps))
    if 0.5 < h/pixels_ps < 0.7:
        print("Type car: Hatchback")
    elif 0.7 < h/pixels_ps < 0.9:
        print("Type car: Sedan")
    elif 0.9 < h/pixels_ps < 1.0:
        print("Type car: SUV")
    elif h/pixels_ps > 1.0:
        print("Type car: PickupTruck")