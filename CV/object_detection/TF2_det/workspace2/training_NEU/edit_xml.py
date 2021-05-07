"""

correct xml files with missing file extension

"""
import os
import pandas as pd

file_loc = "C:/Users/Desktop/data/NEU_detection_efficientdet/voc2coco-master/correct_xml_files/"
files = os.listdir(file_loc)
file_list = [f for f in files if f.endswith('.xml')]

for file in file_list:
    file = file_loc + file
    f = open(file)
    data = f.readlines()
    data_new=data.splitlines()
    
    
    
'''
## edit the output csv

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for member in root.findall('object'):
            
            name = root.find('filename').text
            if '.jpg' in name:
                name = name
            else:
                name = name+'.jpg'
            value = (name, #root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = "C:/Users/DAR9KOR/Desktop/data/sample_datasets/defect_detection/2_codes/NEU_detection_efficientdet/voc2coco-master/correct_xml_files/"
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv((image_path + 'test_labels.csv'), index=None)

main()
'''
