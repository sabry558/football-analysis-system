import cv2
from ultralytics import YOLO
import supervision as sv
import pickle 
import os
import numpy as np
import pandas as pd
from utils import get_center_of_bbox, get_width_of_bbox,get_foot_position
class Tracker:
    """
    A class for tracking objects (players, referee, ball) in a football match video using YOLO and ByteTrack.
    """

    def __init__(self, model_path):
        """
        Initializes the Tracker with a specific YOLO model.

        Args:
            model_path (str): The file path to the YOLO model weights.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self,tracks):
        """
        Calculates and adds the position (center for ball, foot position for others) to the tracking data.

        Args:
            tracks (dict): Dictionary containing object track information.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
    def interpolate_ball_position(self,ball_postion):
        """
        Functions interpolates missing ball positions in the tracking data.

        Args:
            ball_postion (list): A list of dictionaries representing the ball's position across frames.

        Returns:
            list: Interpolated list of ball positions.
        """
        ball_positions=[x.get(1,{}).get('bbox',[]) for x in ball_postion] 
        df_ball_positions=pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
        df_ball_positions=df_ball_positions.interpolate(method='linear',limit_direction='both')
        df_ball_positions=df_ball_positions.bfill()
        ball_positions=[{1:{'bbox':x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
    def detect_frames(self, frames,batch_size=16):
        """
        Detects objects in the video frames using the YOLO model.

        Args:
            frames (list): List of video frames to infer.
            batch_size (int, optional): The batch size for prediction. Defaults to 16.

        Returns:
            list: A list of detection results for each frame.
        """
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            results = self.model.predict(batch_frames,conf=0.1)
            detections+=results
        return detections
  
    def get_object_tracks(self, frames,read_from_stub=False,stub_path=None):
        """
        Gets the object tracks, utilizing ByteTrack upon YOLO detections.

        Args:
            frames (list): Frames to extract tracks from.
            read_from_stub (bool, optional): Flag to load tracking data from file. Defaults to False.
            stub_path (str, optional): The path to the stub pickle file. Defaults to None.

        Returns:
            dict: The dictionary containing structured tracking data for players, ball, and referee.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections=self.detect_frames(frames)

        tracks={
            'player':[],
            'ball':[],
            'referee':[]
        }
        for frame_num,detection in enumerate(detections):
            class_names=detection.names
            class_names_inv={v:k for k,v in class_names.items()}

            detection_supervision=sv.Detections.from_ultralytics(detection)
            for object_ind,class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id]=='goalkeeper':
                    detection_supervision.class_id[object_ind]=class_names_inv['player']
            detections_with_tracks=self.tracker.update_with_detections(detection_supervision)        

            tracks['ball'].append({})
            tracks['player'].append({})
            tracks['referee'].append({})
            for track in detections_with_tracks:
                bbox=track[0].tolist()
                cls_id=track[3]
                track_id=track[4]
                if class_names_inv[cls_id]=='player':
                    tracks['player'][frame_num][track_id]={'bbox':bbox}
                elif class_names_inv[cls_id]=='referee':
                    tracks['referee'][frame_num][track_id]={'bbox':bbox}       

            for track in detection_supervision:
                bbox=track[0].tolist()
                cls_id=track[3]
                track_id=track[4]
                if class_names_inv[cls_id]=='ball':
                    tracks['ball'][frame_num][track_id]={'bbox':bbox}   
        if stub_path is not None:             
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)
        return tracks                        
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        """
        Draws an ellipse around an object at its base position.

        Args:
            frame (np.ndarray): Target video frame to draw on.
            bbox (list): Bounding box of the object.
            color (tuple): RGB color tuple for the ellipse.
            track_id (int, optional): Identifier to overlay on the object. Defaults to None.

        Returns:
            np.ndarray: Assessed image frame with drawn shapes.
        """
        y2=int(bbox[3])
        center_x,_=get_center_of_bbox(bbox)
        width=get_width_of_bbox(bbox)

        cv2.ellipse(frame,
                    center=(center_x,y2),
                    axes=(width,int(0.35*width)),
                    angle=0,
                    startAngle=-45,
                    endAngle=235,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_4)
        rectangle_width=40
        rectangle_height=20
        x1_rect=int(center_x-rectangle_width//2)
        x2_rect=int(center_x+rectangle_width//2)
        y1_rect=int(y2-rectangle_height//2)+15
        y2_rect=int(y2+rectangle_height//2)+15

        if track_id is not None:
            cv2.rectangle(frame,(int(x1_rect),int(y1_rect)),(int(x2_rect),int(y2_rect)),color,cv2.FILLED)
            cv2.putText(frame,str(track_id),(x1_rect+12,y1_rect+15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
        return frame
    def draw_traingle(self,frame,bbox,color):
        """
        Draws a pointing triangle above the bound box (e.g., ball possessor).

        Args:
            frame (np.ndarray): The target video frame.
            bbox (list): Bounding box of the object.
            color (tuple): RGB color for the generated triangle.

        Returns:
            np.ndarray: Returned frame containing the drawn triangular marking.
        """
        y=int(bbox[1])
        center_x,_=get_center_of_bbox(bbox)
        traingle_points=np.array([[center_x,y],[center_x-10,y-20],[center_x+10,y-20]])
        cv2.drawContours(frame,[traingle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame,[traingle_points],0,(0,0,0),2)

        return frame 
    
    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        """
        Overlays the team ball possession statistics onto a frame.

        Args:
            frame (np.ndarray): Renderable video frame.
            frame_num (int): Specific frame count in question.
            team_ball_control (np.ndarray): Sequence recording team possession per frame.

        Returns:
            np.ndarray: Modified given frame resolving rendered text statistics.
        """
        overlay=frame.copy()
        cv2.rectangle(overlay,(1350,850),(1900,970),(255,255,255),-1)
        alpha=0.4
        cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)
        team_ball_control_till_frame=team_ball_control[:frame_num+1]
        team_1_num_frames=team_ball_control_till_frame[team_ball_control_till_frame=='team_1'].shape[0]
        team_2_num_frames=team_ball_control_till_frame[team_ball_control_till_frame=='team_2'].shape[0]
        team_1=team_1_num_frames/(team_1_num_frames+team_2_num_frames+1e-5)
        team_2=team_2_num_frames/(team_1_num_frames+team_2_num_frames+1e-5)
        
        cv2.putText(frame,f'Team 1 Ball Control: {team_1*100:.2f}%',(1400,900),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        cv2.putText(frame,f'Team 2 Ball Control: {team_2*100:.2f}%',(1400,950),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        return frame

    def draw_annotations(self,video_frames,tracks,team_ball_control):
        """
        Applies necessary tracking outlines such as circles and triangles on valid items.

        Args:
            video_frames (list): Unmodified video frames sequence.
            tracks (dict): Frame by frame tracking dictionary per identified entities.
            team_ball_control (np.ndarray): Array determining possession data across time.

        Returns:
            list: Contains all the newly-rendered sequence of visually-altered frames.
        """
        output_frames=[]
        for frame_num,frame in enumerate(video_frames):
            frame=frame.copy()

            player_tracks=tracks['player'][frame_num]
            referee_tracks=tracks['referee'][frame_num] 
            ball_tracks=tracks['ball'][frame_num]
            for track_id,player in player_tracks.items():
                player_team_color=player.get('team_color',(0,255,255))
                frame=self.draw_ellipse(frame,player['bbox'],player_team_color,track_id)
                if player.get('has_ball',False):
                    frame=self.draw_traingle(frame,player['bbox'],(0,0,255))

            for _,referee in referee_tracks.items():
                frame=self.draw_ellipse(frame,referee['bbox'],(255,0,0))  
            for _,ball in ball_tracks.items():
                frame=self.draw_traingle(frame,ball['bbox'],(0,255,0))
            frame=self.draw_team_ball_control(frame,frame_num,team_ball_control)    
            output_frames.append(frame)
        return output_frames        