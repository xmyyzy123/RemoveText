import glob
import os
import argparse
import json
import xml.etree.ElementTree as ET
import json
import shutil
import random

xml_img_path = 'path/to/your/xml jpg/folder'
paddleocr_root_path = 'path/to/your/generating/paddleocr/format/folder'
if not os.path.exists(paddleocr_root_path):
    os.makedirs(paddleocr_root_path)

ratio = 1.0 # train / val

def xml_reader(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    xml_name = tree.find('filename').text
    # size = tree.find('size')
    # width = int(size.find('width').text)
    # height = int(size.find('height').text)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)
    return xml_name, objects

def copyFile():
    # make train and val directory
    train_dir = os.path.join(paddleocr_root_path, "train")
    val_dir = os.path.join(paddleocr_root_path, "val")
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(val_dir)

    # shuffle
    random.seed()
    images = glob.glob(xml_img_path + '/*.jpg')
    filenumber = len(images)
    picknumber = int(filenumber * ratio)
    random.shuffle(images)

    # copy file
    for name in images[:picknumber]:
        shutil.copy(name, os.path.join(train_dir, name.split('/')[-1]))
    if filenumber == picknumber:
        print("ratio is too big. All train dataset will be as val.")
        for name in images[:]:
            shutil.copy(name, os.path.join(val_dir, name.split('/')[-1]))
    else:
        for name in images[picknumber:]:
            shutil.copy(name, os.path.join(val_dir, name.split('/')[-1]))
    print("Copy file done.")
    return images[:picknumber], images[picknumber:] if filenumber != picknumber else images

def write_paddleocr_txt(train_data, val_data):
    """  Write labels into txt which is paddleocr format """
    # write train_label.txt
    with open(os.path.join(paddleocr_root_path, 'train_label.txt'), "w") as f:
        for img_name in train_data:
            label = []
            xml_name = img_name.replace('.jpg', '.xml')
            name, objects = xml_reader(xml_name)
            for obj in objects:
                box = obj['bbox']
                text_name = obj['name'] # Warning: use label as text, because we only use this model to detect
                lt = [box[0], box[1]] # left top
                rt = [box[2], box[1]] # right top
                rb = [box[2], box[3]] # right bottom
                lb = [box[0], box[3]] # left bottom
                s = [lt, rt, rb, lb]
                result = {"transcription": text_name, "points": s}
                label.append(result)

            f.write("train/" + name + '\t' + json.dumps(
                label, ensure_ascii=False) + '\n')

    # write val_label.txt
    with open(os.path.join(paddleocr_root_path, 'val_label.txt'), "w") as f:
        for img_name in val_data:
            label = []
            xml_name = img_name.replace('.jpg', '.xml')
            name, objects = xml_reader(xml_name)
            for obj in objects:
                box = obj['bbox']
                text_name = obj['name'] # Warning: use label as text, because we only use this model to detect
                lt = [box[0], box[1]] # left top
                rt = [box[2], box[1]] # right top
                rb = [box[2], box[3]] # right bottom
                lb = [box[0], box[3]] # left bottom
                s = [lt, rt, rb, lb]
                result = {"transcription": text_name, "points": s}
                label.append(result)

            f.write("val/" + name + '\t' + json.dumps(
                label, ensure_ascii=False) + '\n')
    
    print("Write txt done.")

def gen_paddleocr_data():
    """  Generate paddleocr dataset """
    train_data, val_data = copyFile()
    write_paddleocr_txt(train_data, val_data)

if __name__ == "__main__":
    gen_paddleocr_data()
    print("Finish!")