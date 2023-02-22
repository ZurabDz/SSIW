import cv2
import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass
class Point:
    x1: float
    x2: float
    y1: float
    y2: float


@dataclass
class ObjectMetadata:
    points: Point
    label: int
    label_name: str
    flag: int


def parse_cmp_string_xml(chunk: list):
    et = ET.fromstring(' '.join(chunk))

    points, label, labelname, flag = list(et)
    points = [float(point.text.strip()) for point in points]
    label = int(label.text.strip())
    labelname = labelname.text.strip()
    flag = int(flag.text.strip())

    return ObjectMetadata(Point(*points), label, labelname, flag)


def parse_cmp_xml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    metadatas = []
    chunk = []

    for element in data:
        if element.startswith('</object>'):
            chunk.append(element)
            metadata = parse_cmp_string_xml(chunk)
            chunk.clear()
            metadatas.append(metadata)
        else:
            chunk.append(element)

    return metadatas


metadatas = parse_cmp_xml('/home/penguin/SSIW/data/base/cmp_b0001.xml')

path = r'/home/penguin/SSIW/data/base/cmp_b0001.jpg'
image = cv2.imread(path)
height, width, _ = image.shape
print(width, height)

print(metadatas[0])


a = 0
for i, obj in enumerate(metadatas):
    # if obj.label_name != 'window':
        # continue
    obj.points.x1 *= height
    obj.points.x2 *= height
    obj.points.y1 *= width
    obj.points.y2 *= width

    start_point = (round(obj.points.y1), round(obj.points.x1))
    end_point = (round(obj.points.y2), round(obj.points.x2))
    color = (255, 0, 0)
    thickness = 1
    # print('ssss: ', obj)
    # image = cv2.rotate(image, cv2.ROTATE_180)
    cv2.rectangle(image, start_point, end_point, color, thickness)
    # break

# print(a)
cv2.imwrite("./test.png", image)
# cv2.imshow('test', image)
# cv2.waitKey(0)
# # and finally destroy/close all open windows
# cv2.destroyAllWindows()


