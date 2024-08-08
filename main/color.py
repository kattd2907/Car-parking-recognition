import cv2
import os
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier

image_dir = r'C:\Users\Admin\PycharmProjects\pythonProjectest\image_crop'

path = r'C:\Users\Admin\PycharmProjects\pythonProjectest\training.data'
if os.path.isfile(path) and os.access(path, os.R_OK):
    print('training data is ready, classifier is loading')
else:
    print('training data is being created')
    open(path, 'w')
    color_histogram_feature_extraction.training()
    print('training data is ready, classifier is loading')

file_list = sorted(os.listdir(image_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
for filename in file_list:
    image_path = os.path.join(image_dir, filename)
    source_image = cv2.imread(image_path)

    color_histogram_feature_extraction.color_histogram_of_test_image(source_image)
    prediction = knn_classifier.main(path, r'C:\Users\Admin\PycharmProjects\pythonProjectest\main\test.data')

    print(f'{filename}: Mau sac: {prediction}')

    cv2.putText(
        source_image,
        'Color: ' + prediction,
        (15, 45),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (255, 0, 0),
    )

    cv2.imshow('color classifier', source_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
