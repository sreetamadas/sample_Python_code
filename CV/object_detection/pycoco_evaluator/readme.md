https://github.com/cocodataset/cocoapi/tree/master/PythonAPI   <- evaluator
Convert GT (http://cocodataset.org/#format-data ) & detections (http://cocodataset.org/#format-results ) to json files
The box inputs have to be in the format (x y w h)
Both GT & predictions have to be in specified json file format
A single json file was prepared from the prediction outputs, of individual frames, in text files
Ground truth json was created from the GT annotation file (.top -> .csv -> .json)


Evaluation tools:
1. pycoco evaluator

2. https://github.com/rafaelpadilla/Object-Detection-Metrics#asterisk  
considered only people predictions (output in the form xyrb, r: right, b: bottom)
GT box annotation has - class, x_centre, y_centre, w, h  (as fractional coordinates), and were converted to actual coordinates & checked by plotting on the images
Evaluation tool requires GT & predictions from each image as separate text files

3. https://towardsdatascience.com/evaluating-performance-of-an-object-detection-model-137a349c517b   (PASCAL VOC metric implementation used) 
considered only people predictions (output as xywh in separate txt file for each frame) 
– these were converted to (x y x+w y+h) format & saved in a single csv file, for use with the evaluation tool
