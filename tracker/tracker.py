from ultralytics import YOLO
import supervision as sv
import pickle 
import os
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