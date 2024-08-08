import cv2
import numpy as np
import math

start_point = None
end_point = None
drawing = False


def draw_line(event, x, y, flags, param):
    global start_point, end_point, drawing, image

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        drawing = True

    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        drawing = False
        cv2.line(image, start_point, end_point, (255, 255, 0), 2)


        length = math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
        print(f"Chieu dai: {length} pixels")

image = cv2.imread(r'C:\Users\Admin\PycharmProjects\pythonProjectest\image_crop\car_1.jpg')

cv2.namedWindow('Line Drawer')
cv2.setMouseCallback('Line Drawer', draw_line)

while True:
    cv2.imshow('Line Drawer', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
