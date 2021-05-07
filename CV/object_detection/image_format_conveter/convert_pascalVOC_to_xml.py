import os

import xml.etree.ElementTree as ET
from xml.dom import minidom


def convertor_f(filename, img_dim, classnames, bboxes, save_path,
                file_extn=".xml"):
    """
    Convert pascalvoc data to xml format.

    Parameters
    ----------
    filename : str
        Image filename
    img_dim : str
       Image size(height,width,depth)
    classnames : list
        List of classes in a given image
    bboxes : list
        list of lists with bounding box coordinates
    save_path : str
        Path to output xml
    file_extn : str, optional. The default is ".xml".

    Returns
    -------
    None.

    """
    xml_name = os.path.split(filename)[1].split('.')[0]
    root_node = ET.Element('annotation')
    folder_node = ET.SubElement(root_node, 'folder')
    folder_node.text = os.path.split(filename)[1].split('_')[0]
    filename_node = ET.SubElement(root_node, 'filename')
    filename_node.text = os.path.split(filename)[1]
    source_node = ET.SubElement(root_node, 'source')
    database_node = ET.SubElement(source_node, 'database')
    database_node.text = "NEU-DET"
    size_node = ET.SubElement(root_node, 'size')
    width_child_node = ET.SubElement(size_node, 'width')
    width_child_node.text = str(img_dim[0])
    height_child_node = ET.SubElement(size_node, 'height')
    height_child_node.text = str(img_dim[1])
    depth_child_node = ET.SubElement(size_node, 'depth')
    depth_child_node.text = str(img_dim[2])
    segmented_node = ET.SubElement(root_node, 'segmented')
    for count in range(len(bboxes)):
        object_node = ET.SubElement(root_node, 'object')
        name_node = ET.SubElement(object_node, 'name')
        name_node.text = classnames[count]
        pose_node = ET.SubElement(object_node, 'pose')
        pose_node.text = "Unspecified"
        truncated_node = ET.SubElement(object_node, 'truncated')
        truncated_node.text = "1"
        difficult_node = ET.SubElement(object_node, 'difficult')
        difficult_node.text = "0"
        bnd_box_node = ET.SubElement(object_node, 'bndbox')
        xmin_node = ET.SubElement(bnd_box_node, 'xmin')
        xmin_node.text = str(bboxes[count][0])
        ymin_node = ET.SubElement(bnd_box_node, 'ymin')
        ymin_node.text = str(bboxes[count][1])
        xmax_node = ET.SubElement(bnd_box_node, 'xmax')
        xmax_node.text = str(bboxes[count][2])
        ymax_node = ET.SubElement(bnd_box_node, 'ymax')
        ymax_node.text = str(bboxes[count][3])
    xmlfile = open(os.path.join(save_path, xml_name + file_extn), "wb")
    rough_string = ET.tostring(root_node)
    reparsed = minidom.parseString(rough_string)
    xmlfile.write(reparsed.toprettyxml(encoding='utf8'))


def get_bbox_count_f(xml_file_path):
    """
    Compute the number of bounding boxes present in an image via xml.

    Parameters
    ----------
    xml_file_path : str
        Path to XML file

    Returns
    -------
    bbox_count : int
        Number of bounding boxes

    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    bbox_count = len([elem for elem in root.iterfind('./object')])
    return bbox_count
