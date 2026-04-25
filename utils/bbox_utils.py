def get_center_of_bbox(bbox):
    x1,y1,x2,y2=bbox
    center_x=int((x1+x2)/2)
    center_y=int((y1+y2)/2)
    return center_x,center_y

def get_width_of_bbox(bbox):
    x1,y1,x2,y2=bbox
    width=x2-x1
    return width

def calc_distance(point1,point2):
    distance=((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)**0.5
    return distance
def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]
def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)