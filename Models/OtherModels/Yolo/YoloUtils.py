import numpy as np


def get_y_true(img, label, gt_coords, num_classes, feature_layer_dims, anchors, threshold=0.5):
    '''
    Inputs-: 
            Image for which y true needs to be returned of shape (?,?,3)
            Label for the image - int
            Groundth truth box coordinates - python list (format [cx,cy,w,h] all in range [0,1])  
            Number of classes(excluding background)- int
            List of size of feature layer sides - python list 
            Size of anchor boxes per layer - Numpy array of shape (num_layers,2) with (w,h) in last axis
            
    Returns a (?,num_classes+5) numpy vector which reresents ground truth vector for loss calculation
    '''
    
    width,height = img.shape[1],img.shape[0]
    
    all_boxes = []  # array to house all boxes
    
    
    y_true = []
    
    for fh,fw in feature_layer_dims:
        y_true.append(np.zeros((fh,fw,(5+num_classes))))
    
    for i,(fh,fw) in enumerate(feature_layer_dims):
        
        x = np.linspace(0,fw-1,fw)
        y = np.linspace(0,fh-1,fh)
        
        x,y = np.meshgrid(x,y)
        
        w,h = np.ones((fh,fw))*anchors[i,0], np.ones((fh,fw))*anchors[i,1]
        
        anchor_boxes = np.stack([x,y,w,h],axis=-1)
        
        ious = iou([(x+0.5)/fw,(y+0.5)/fh,w,h],gt_coords)
        
        y_true[i][ious>threshold,4] = -1 
    
        all_boxes.append(anchor_boxes)
    
    
    best = best_box(anchors,gt_coords)
    
    best_h,best_w = feature_layer_dims[best]
    
    gx,gy,gw,gh = gt_coords
    
    i,j = int(gy*best_h),int(gx*best_w)
    y_true[best][i,j,4] = 1         #objectness =1
    y_true[best][i,j,label + 5] = 1  # class score = 1
    
    def sinv(y):
        y = max(y,1e-15)
        y = min(1-1e-15,y)
        return np.log(y/(1-y))
    
    y_true[best][i,j,0] = sinv(gx*best_w -j)
    y_true[best][i,j,1] = sinv(gy*best_h -i)
    y_true[best][i,j,2] = np.log(gw/anchors[best,0])
    y_true[best][i,j,3] = np.log(gh/anchors[best,1])
    
    return y_true

def iou(anchor_boxes, gt_coords):
    '''
    Return vector with ious of every anchor box with ground truth box
    Arguments-:
    List of Numpy array with anchor box coordinates [cx,cy,w,h]
    List/Numpy array with ground truth coords (cx,cy,w,h)
    '''
    
    cx,cy,w,h = anchor_boxes
    x1,y1,x2,y2 = cx-w/2,cy-h/2,cx+w/2,cy+h/2  # convert to corner point format
    
    gx,gy,gw,gh = gt_coords
    
    gx1,gy1,gx2,gy2 = gx-gw/2,gy-gh/2,gx+gw/2,gy+gh/2
    
    #intesection cordinates
    xA = np.maximum(x1, gx1)
    yA = np.maximum(y1, gy1)
    xB = np.minimum(x2, gx2)
    yB = np.minimum(y2, gy2)
    
    interArea = np.maximum((xB - xA ), 0) * np.maximum((yB - yA ), 0)
    
    abox_area = (x2-x1)*(y2-y1)
    
    gt_area = (gx2-gx1)*(gy2-gy1)
    
    iou = interArea / (abox_area + gt_area - interArea)
    
    return iou

def best_box(anchor_boxes,gt_coords):
    '''
    Arguments-:
    Groundth truth box coordinates - python list (format [cx,cy,w,h] all in range [0,1])
    Size of anchor boxes per layer - Numpy array of shape (num_layers,2) with (w,h) in last axis    
    
    Returns index of layer with best box
    '''
    
    _,_,gw,gh = gt_coords
    
    w,h = anchor_boxes[:,0],anchor_boxes[:,1]
    
    wf = np.minimum(gw,w)
    hf = np.minimum(gh,h)
    
    inter_area = wf*hf
    
    union_area = w*h + gw*gh - inter_area
    
    return np.argmax(inter_area/union_area)