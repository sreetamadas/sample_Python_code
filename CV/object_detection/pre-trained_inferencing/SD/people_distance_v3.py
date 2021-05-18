
import time
import numpy as np
import cv2 
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import Counter
# from numba import jit

def bb2array(res,fieldnames = ["label", "score", "ymin", "ymax", "xmin", "xmax"]):
    res_array = np.array([np.array([item.get(key, '') for key in fieldnames],dtype='float') for item in res])
    return res_array

def a12(ds,pt,x):
    p = (pt-0.5)*x[1]
    radius = 0.5*(ds*np.sin(x[2]-p)**2)/(x[0]*np.tan(x[1]/2)*np.cos(p)**2)
    return radius

def vertical(pt,x,h,ds=6):    
    p = (pt-0.5)*x[1]
    # print('p')
    # print(x[2]-p)
    f = 0.25*h*ds*np.sin(x[2]-p)/(x[0]*np.tan(0.5*x[1])*np.cos(p))
    # f = 0.25*h*(ds-2*np.sin(x[2]-p))*np.sin(x[2]-p)/(x[0]*np.tan(0.5*x[1])*np.cos(p))
    f1 = 0.25*h*ds*np.sin(x[2]-p)*np.cos(x[2]-p)/(x[0]*np.tan(0.5*x[1])*np.cos(p)**2)
    # print(f,f1)
    return f
    

  
def get_ellipses(res,x,h,ds=6):
    # try:
    min_px = 5
    pht = 1 - res[:,2:4]/h
    # p = (pht[:,1]-0.5)*x[1]
    cx = (0.5*np.sum(res[:,4:6],axis=1)).astype(int)
    cy = res[:,3].astype(int)
    r = a12(ds,pht[:,1],x)
    a = vertical(pht[:,1],x,h).astype(int)
    a = np.maximum(a,min_px).astype(int)
    b = np.maximum(0.25*h*r,min_px).astype(int)
    b = np.minimum(a,np.maximum(0.5*h*r,min_px)).astype(int)
    return(pht,cx,cy,a,b)
    # except:
    #     return(None,None,None,None)


def distancing(x,res,ellipses,ds,w,h):
    try:
        # p = (ellipses[0][:,1]-0.5)*x[1]
        # ha=5.6
        # red_bounds,linkages = [],[]
        le = np.expand_dims(ellipses[1]-ellipses[3], axis=0)
        re = np.expand_dims(ellipses[1]+ellipses[3], axis=0)
        ue = np.expand_dims(ellipses[2]-ellipses[4], axis=0)
        be = np.expand_dims(ellipses[2]+ellipses[4], axis=0)
        # l = len(res)
    
        x1 = np.maximum(le,le.T)
        x2 = np.minimum(re,re.T)
        y1 = np.maximum(ue,ue.T)
        y2 = np.minimum(be,be.T)
        inter_width = x2-x1
        inter_height = y2-y1
        # print(x1,x2,y1,y2)
        overlap = (inter_width>0)*(inter_height>0)
        # print(overlap)
        # overlap = np.triu(overlap,1)
        cx = np.expand_dims(ellipses[1], axis=0)
        cy = np.expand_dims(ellipses[2], axis=0)
        a = np.expand_dims(ellipses[3], axis=0)
        b = np.expand_dims(ellipses[4], axis=0)
        a2 = a+a.T
        b2 = b+b.T
        scale = a2/b2
        # union_centre = (0.5*(x1+x2),0.5*(y1+y2))
        dx = cx-cx.T
        dy = scale*(cy-cy.T)
        d2 = np.sqrt(dx**2+dy**2)
        ds2 = np.sqrt(a2**2+b2**2)
        close_dist = overlap*(d2<0.9*ds2)
        np.fill_diagonal(close_dist, False)
        violations = int(np.sum(close_dist)/2)
        graph = csr_matrix(close_dist)
        # print(graph)
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        groups = Counter(labels)
        plural_groups = {k: v for k, v in groups.items() if v >1}
        # print(plural_groups,len(plural_groups))
        # close_dist = np.triu(close_dist,1)
        red_bounds = (np.sum(close_dist,axis=0)>0).nonzero()[0]
    
        return red_bounds,plural_groups,violations
    except:
        return []


def bounding_ellipse(img,x,res,ellipses,w,h,red_bounds):
    # try:
    # edges = 12
    img_new1 = img.copy()
    img_new2 = img.copy()
    # img_new3 = img.copy()
    # overlayed_img = img.copy()
    for i in range(len(res)): 
        if i in red_bounds:
            clr = (0,0,200)
        else:
            clr = (0,200,0)

        # cv2.rectangle(img_new1, (ellipses[1][i]-ellipses[3][i],ellipses[2][i]-ellipses[4][i]), (ellipses[1][i]+ellipses[3][i],ellipses[2][i]+ellipses[4][i]), clr, 2)
        cv2.ellipse(img_new1, (ellipses[1][i],ellipses[2][i]), (ellipses[3][i],ellipses[4][i]), 0, 0, 360, clr, -1)

    alpha = 0.3  # Transparency factor.
    
    # # Following line overlays transparent rectangle over the image
    img_new2 = cv2.addWeighted(img_new1, alpha, img_new2, 1 - alpha, 0)
    
    # add blur
    # frame[x1:x2,y1:y2] = cv2.GaussianBlur(frame[x1:x2,y1:y2],(size,size),cv2.BORDER_DEFAULT)  
    #img_new2 = add_blur(res, img_new2)

    return img_new2
    # except:
    #     print('no bounding box found')
    #     return img

#def add_blur(res,img_new2):
    

# def bounding_ellipse(img,x,res,ellipses,w,h,red_bounds):
#     try:
#         # edges = 12
#         img_new1 = img.copy()
#         img_new2 = img.copy()
#         # img_new3 = img.copy()
#         # overlayed_img = img.copy()
#         for i in range(len(res)): 
#             if i in red_bounds:
#                 clr = (0,0,200)
#             else:
#                 clr = (0,200,0)
    
#             # cv2.rectangle(img_new1, (ellipses[1][i]-ellipses[3][i],ellipses[2][i]-ellipses[4][i]), (ellipses[1][i]+ellipses[3][i],ellipses[2][i]+ellipses[4][i]), clr, 2)
#             cv2.ellipse(img_new1, (ellipses[1][i],ellipses[2][i]), (ellipses[3][i],ellipses[4][i]), 0, 0, 360, clr, -1)
    
#         alpha = 0.3  # Transparency factor.
        
#         # # Following line overlays transparent rectangle over the image
#         img_new2 = cv2.addWeighted(img_new1, alpha, img_new2, 1 - alpha, 0)
    
#         return img_new2
#     except:
#         print('no bounding box found')
#         return img
