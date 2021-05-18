
import cv2
import numpy as np

def add_mask(frame,res1):
    '''logic for blur box: don't use full bounding box
    In x-direction, it is half of box-width about centre -> box moves; take full
    In y-direction, it is 1/3 box height from top
    '''
    size = 99
    #print(res1.shape)
    for i in range(res1.shape[0]):
        x1 = int(res1[i,4]) #int((3*res1[i,4] + res1[i,5])/4)  #int(res1[i,4]) ; instead of xmin, use (3xmin + xmax)/4
        x2 = int(res1[i,5]) #int((res1[i,4] + 3*res1[i,5])/4)  #int(res1[i,5])
        y1 = int(res1[i,2])
        y2 = int((2*res1[i,2] + res1[i,3])/3)  #int(res1[i,3])
        #print(y1,x1,y2,x2)  # this is correct
        
        #frame[x1:x2, y1:y2] = cv2.GaussianBlur(frame[x1:x2, y1:y2],(size,size),cv2.BORDER_DEFAULT)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2],(size,size),cv2.BORDER_DEFAULT)
        # the above step takes y-coord first, then x; otherwise the blurred box is 90deg rotated
    return frame

def occlusion_correction(frame,frame_width,frame_height,res,model='hw_ratio'):
    # print('result before occlusion correction : ',res)
    frame1 = frame.copy()
    updated_res = []
    # start = time.time()
    for i in range(len(res)): 
        ymin = int(res[i]['ymin'])
        xmin = int(res[i]['xmin'])
        ymax = int(res[i]['ymax'])
        xmax = int(res[i]['xmax'])
        bb_w = xmax-xmin
        bb_h = ymax-ymin
        if bb_h<1.6*bb_w:
            # print ('checking for occlusion',bb_h,bb_w)
            # try:
            # cropped_img = frame1[ymin:ymax,xmin:xmax,:]
            if model == 'hw_ratio':
                # print ('correcting for occlusion')
                if bb_h<1.2*bb_w:
                    ymax += int(1*bb_h)
                    res[i]['ymax'] = ymax
                else:
                    ymax += int(0.6*bb_h)
                    res[i]['ymax'] = ymax
                
            else:
                cropped_img = frame1[max(0,ymin):min(frame_height-1,ymax),max(0,xmin):min(frame_width-1,xmax),:]
                cropped_img = Image(pil2tensor(cropped_img, dtype=np.float32).div_(255)) 
                pred,idx,outputs = model.predict(cropped_img)
                if idx==0:
                    print ('correcting for occlusion')
                    ymax += int(0.5*bb_h)
                    res[i]['ymax'] = ymax
                    clr = (255,255,255)
                    frame = cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), clr, 2)
        if res[i]['ymax']<frame_height:
            updated_res.append(res[i])
    # print ('classification time : ',time.time()-start)
                # cv2.imshow('frame',frame)
            # except:
            #     print('Exception')
            #     pass
    # print('result after occlusion correction : ',res)
    return updated_res
