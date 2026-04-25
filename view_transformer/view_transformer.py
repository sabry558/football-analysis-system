import numpy as np 
import cv2

class ViewTransformer():
    """
    Transforms pixel coordinates from a 2D camera perspective into a 2D top-down view 
    (birds-eye view) using perspective transformation.
    """
    def __init__(self):
        """
        Initializes the ViewTransformer by calculating the perspective transformation 
        matrix from predefined pixel vertices to target court vertices.
        """
        court_width = 68
        court_length = 23.32

        self.pixel_vertices = np.array([[110, 1035], 
                               [265, 275], 
                               [910, 260], 
                               [1640, 915]])
        
        self.target_vertices = np.array([
            [0,court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self,point):
        """
        Applies perspective transformation to a single point.

        Args:
            point (list or tuple or numpy.ndarray): The (x, y) coordinates of the point.

        Returns:
            numpy.ndarray or None: The transformed (x, y) coordinates, or None if the point 
                                   lies outside the polygon defined by `pixel_vertices`.
        """
        p = (int(point[0]),int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0 
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        tranform_point = cv2.perspectiveTransform(reshaped_point,self.persepctive_trasnformer)
        return tranform_point.reshape(-1,2)

    def add_transformed_position_to_tracks(self,tracks):
        """
        Iterates over all object tracks and adds their transformed coordinates.

        Args:
            tracks (dict): Dictionary comprising frame-by-frame object tracks.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_trasnformed = self.transform_point(position)
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed