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



metadatas = parse_cmp_xml('path to cmp facade xml file')


