from hope.utils.xml_utils import parse_cmp_xml
import cv2

metadatas = parse_cmp_xml('/home/penguin/SSIW/data/base/cmp_b0002.xml')

path = r'/home/penguin/SSIW/data/base/cmp_b0002.jpg'
image = cv2.imread(path)

height, width, _ = image.shape
print(width, height)
print(metadatas[0])


for i, obj in enumerate(metadatas):
    obj.points.x1 *= height
    obj.points.x2 *= height
    obj.points.y1 *= width
    obj.points.y2 *= width

    start_point = (round(obj.points.y1), round(obj.points.x1))
    end_point = (round(obj.points.y2), round(obj.points.x2))
    color = (255, 0, 0)
    thickness = 1
    cv2.rectangle(image, start_point, end_point, color, thickness)

cv2.imwrite("./test.png", image)
# cv2.imshow('test', image)
# cv2.waitKey(0)
# # and finally destroy/close all open windows
# cv2.destroyAllWindows()