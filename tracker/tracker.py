import cv2
from ultralytics import YOLO
import supervision as sv
import pickle 
import os
import numpy as np
from utils import get_center_of_bbox, get_width_of_bbox
class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames,batch_size=16):
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            results = self.model.predict(batch_frames,conf=0.1)
            detections+=results
        return detections

  
    def get_object_tracks(self, frames,read_from_stub=False,stub_path=None):
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
        y=int(bbox[1])
        center_x,_=get_center_of_bbox(bbox)
        traingle_points=np.array([[center_x,y],[center_x-10,y-20],[center_x+10,y-20]])
        cv2.drawContours(frame,[traingle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame,[traingle_points],0,(0,0,0),2)

        return frame 

    def draw_annotations(self,video_frames,tracks):
        output_frames=[]
        for frame_num,frame in enumerate(video_frames):
            frame=frame.copy()

            player_tracks=tracks['player'][frame_num]
            referee_tracks=tracks['referee'][frame_num] 
            ball_tracks=tracks['ball'][frame_num]
            for track_id,player in player_tracks.items():
                frame=self.draw_ellipse(frame,player['bbox'],(0,0,255),track_id)

            for _,referee in referee_tracks.items():
                frame=self.draw_ellipse(frame,referee['bbox'],(255,0,0))  
            for _,ball in ball_tracks.items():
                frame=self.draw_traingle(frame,ball['bbox'],(0,255,0))
            output_frames.append(frame)
        return output_frames        