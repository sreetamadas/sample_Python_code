# convert GT coordinates from fractions, in yolo format , to pixel coordinates



######################################################################################

def correct_gt(photo_filename, gt_name, out_name, file_j):
  #load and read image
  image = cv2.imread(photo_filename)
  (H, W) = image.shape[:2]

  # create output file for scaled ground truths
  outf = open(out_name, 'a')

  # read in ground truth
  FileIN = open(gt_name, 'r')  

  for Line in FileIN:
    string_data = Line.split(sep=' ') 
    string_to_float = [float(i) for i in string_data]     
    
    # correct scale & write out to new file
    # scale the bounding box coordinates back relative to the
    # size of the image, keeping in mind that YOLO actually
    # GT annotation has - class, x_centre, y_centre, w, h  (as fractions)
    box = string_to_float[1:5] * np.array([W, H, W, H])
    (centerX, centerY, width, height) = box.astype("int")
    
    # use the center (x, y)-coordinates to derive the top and
    # and left corner of the bounding box
    x1 = int(centerX - (width / 2))
    y1 = int(centerY - (height / 2))

    #(x1, y1, x2, y2) = box.astype("int")
    s = str(int(string_to_float[0])) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(x1+width) + ' ' + str(y1+height) + "\n"
    outf.write(s)


    #'''
    # check this is working correctly by plotting on to image
    # draw a bounding box rectangle and label on the image
    # define the labels
    LABELS = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
                  
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    color = [int(c) for c in COLORS[0]]
    cv2.rectangle(image, (x1, y1), (x1+width, y1+height), color, 2)

  out_img = "/content/ground_truth_img/" + file_j #"/content/det_yolo/ground_truth_yolo/test_out.jpg" #out_path + name + file_no[j]
  cv2.imwrite(out_img, image)
  #'''
  
  outf.close
  FileIN.close()

#####################################################################


#out_path = "/content/ground_truth/"
#out_img_path = "/content/ground_truth_img/"
image_paths = "/content/img/"
gt_path = "/content/gt/"

file_no = os.listdir(image_paths)

for j in range(0, len(file_no)): 
  print(file_no[j])
  photo_filename = image_paths + file_no[j]
  gt_name = file_no[j].split('.jpg')[0]
  out_name = out_path + gt_name + '.txt'
  gt_name = gt_path + gt_name + '.txt'
  correct_gt(photo_filename, gt_name, out_name, file_no[j])





##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

# check GT after scaling

img_path =  "/content/img/"
out_img_path = "/content/ground_truth_img/"
out_path = "/content/ground_truth/"



def correct_gt(photo_filename, out_name, file_j):
  #load and read image
  image = cv2.imread(photo_filename)
  (H, W) = image.shape[:2]

  # read in ground truth
  FileIN = open(out_name, 'r')  

  for Line in FileIN:
    string_data = Line.split(sep=' ') 
    string_to_int = [int(i) for i in string_data] 
    x1 = string_to_int[1]
    y1 = string_to_int[2]
    x2 = string_to_int[3]
    y2 = string_to_int[4]
    
    #'''
    # check this is working correctly by plotting on to image
    # draw a bounding box rectangle and label on the image
    # define the labels
    LABELS = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
                  
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    color = [int(c) for c in COLORS[0]]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
  out_img = "/content/ground_truth_img/" + file_j #"/content/det_yolo/ground_truth_yolo/test_out.jpg" #out_path + name + file_no[j]
  cv2.imwrite(out_img, image)
  #'''
  
  #outf.close
  FileIN.close()




file_no = os.listdir(img_path)

for j in range(0, len(file_no)): 
  print(file_no[j])
  photo_filename = img_path + file_no[j]
  gt_name = file_no[j].split('.jpg')[0]
  out_name = out_path + gt_name + '.jpg.txt'
  #gt_name = gt_path + gt_name + '.txt'
  correct_gt(photo_filename, out_name, file_no[j])


