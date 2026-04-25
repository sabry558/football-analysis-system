def get_center_of_bbox(bbox):
    """
    Calculates the center (x, y) coordinates of a bounding box.

    Args:
        bbox (list or tuple): The bounding box coordinates (x1, y1, x2, y2).

    Returns:
        tuple: (center_x, center_y) as integers.
    """
    x1,y1,x2,y2=bbox
    center_x=int((x1+x2)/2)
    center_y=int((y1+y2)/2)
    return center_x,center_y

def get_width_of_bbox(bbox):
    """
    Calculates the width of a bounding box.

    Args:
        bbox (list or tuple): The bounding box coordinates (x1, y1, x2, y2).

    Returns:
        int/float: The width of the bounding box.
    """
    x1,y1,x2,y2=bbox
    width=x2-x1
    return width

def calc_distance(point1,point2):
    """
    Calculates the Euclidean distance between two points.

    Args:
        point1 (tuple or list): The first point (x1, y1).
        point2 (tuple or list): The second point (x2, y2).

    Returns:
        float: The Euclidean distance between point1 and point2.
    """
    distance=((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)**0.5
    return distance
def measure_xy_distance(p1,p2):
    """
    Calculates the difference in x and y coordinates between two points.

    Args:
        p1 (tuple or list): The first point (x1, y1).
        p2 (tuple or list): The second point (x2, y2).

    Returns:
        tuple: (dx, dy) which is the difference in x and y components.
    """
    return p1[0]-p2[0],p1[1]-p2[1]
def get_foot_position(bbox):
    """
    Calculates the position of the player's feet (bottom center) from a bounding box.

    Args:
        bbox (list or tuple): The bounding box coordinates (x1, y1, x2, y2).

    Returns:
        tuple: (center_x, y2) as integers representing the foot position.
    """
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)