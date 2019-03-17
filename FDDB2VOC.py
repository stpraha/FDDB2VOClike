import cv2
import numpy as np
import os
import shutil
import xml.etree.ElementTree as ET

img_path = 'F:/FDDB/FDDBpics/'
folds_path = 'F:/FDDB/FDDBfolds/'

out_xml_path = 'F:/FDDB2VOC/Annotations/'
out_pic_path = 'F:/FDDB2VOC/JPEGImages/'

def get_rectangle_from_ellipse(img, boxes):
    rect_boxes = []
    img_w = img.shape[1]
    img_h = img.shape[0]
    for i, box in enumerate(boxes):
        paras = box.split(' ')
        
        major_axis_radius = float(paras[0])
        minor_axis_radius = float(paras[1]) 
        angle = float(paras[2]) / 3.1415926 * 180
        center_x= float(paras[3])
        center_y = float(paras[4])
        #<major_axis_radius minor_axis_radius angle center_x center_y 1>.
        #123.583300 85.549500 1.265839 269.693400 161.781200  1
        cv2.ellipse(img, (int(center_x), int(center_y)), (int(major_axis_radius), int(minor_axis_radius)), int(angle), 0, 360, 255)
        
        mask = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)
        cv2.ellipse(mask, (int(float(center_x)), int(center_y)), (int(major_axis_radius), int(minor_axis_radius)), int(angle), 0, 360, (255, 255, 255))
        image, contonurs, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        #for i in range(len(contonurs)):
            #print(i)
        rec = cv2.boundingRect(contonurs[0])
        xmin = rec[0]
        ymin = rec[1]
        w = rec[2]
        h = rec[3]
        xmax = xmin + w
        ymax = ymin + h
        x_center = xmin + w / 2
        y_center = ymin + h / 2
            
        rect_boxes.append(np.stack([ymin, xmin, ymax, xmax]))
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
    #print(img_w, img_h)
    #print(rect_boxes)
        
    return img, img_w, img_h, rect_boxes

def indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def save_xml(img_name, img_w, img_h, rect_boxes):
    root = ET.Element('annotation')
    tree = ET.ElementTree(root)
    
    folder = ET.SubElement(root, 'folder')
    folder.text = 'FDDB'
    
    filename = ET.SubElement(root, 'filename')
    filename.text = img_name
    
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(img_w)
    height = ET.SubElement(size, 'height')
    height.text = str(img_h)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(3)
    
    for i, box in enumerate(rect_boxes):
        objects = ET.Element('object')
        name = ET.SubElement(objects, 'name')
        name.text = 'face'
        bndbox = ET.SubElement(objects, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(box[1])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(box[0])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(box[3])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(box[2])
        
        root.append(objects)
        
    indent(root)
    xml_name = img_name.split('.')[0] + '.xml'
    xml = out_xml_path + xml_name
    tree.write(xml)

def move_pic(pic_path, img_name):
    new_path = out_pic_path + img_name
    shutil.copyfile(pic_path, new_path)
    

def deal_single_fold_txt(path):
    substr = 'img_'
    with open(path, 'r') as f:
        for line in f:
            if line.find(substr) != -1:
                line = line.replace("\n","")
                pic_path = img_path + line + '.jpg'
                img_name = pic_path.split('/')[-5] + pic_path.split('/')[-4] + pic_path.split('/')[-3] + pic_path.split('/')[-1]
                print(img_name)
                
                move_pic(pic_path, img_name)
                img = cv2.imread(pic_path)
                
                box_num = f.readline().replace("\n","")  
                boxes = []   
                for i in range(int(box_num)):
                    boxes.append(f.readline().replace("\n",""))
            
                img, img_w, img_h, rect_boxes = get_rectangle_from_ellipse(img, boxes)
#                cv2.imshow('image', img) 
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()
                
                save_xml(img_name, img_w, img_h, rect_boxes)
                #break
                
                
def read_fold():
    filelist = os.listdir(folds_path)
    for i, txt_file in enumerate(filelist):
        deal_single_fold_txt(folds_path + txt_file)
        #break

read_fold()