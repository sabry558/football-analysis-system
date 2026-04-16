from team_assigner.team_assigner import teamAssigner
from player_ball_assigner import playerBallAssigner
from utils import read_video, save_video
from tracker import Tracker
import numpy as np
def main():
  video_frames=read_video("input_video/input.mp4")
  tracker=Tracker('model training/model/best.pt')
  tracks=tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path='stub/tracks.pkl')
  tracks['ball']=tracker.interpolate_ball_position(tracks['ball'])
  team_assigner=teamAssigner()
  team_assigner.assign_team_color(video_frames[0],tracks['player'][0])

  for frame_num,player_track in enumerate(tracks['player']):
    for player_id,track in player_track.items():
        player_bbox=track['bbox']
        team_name=team_assigner.get_player_team(video_frames[frame_num],player_bbox,player_id)
        
        tracks['player'][frame_num][player_id]['team_color']=team_assigner.team_color[team_name]
  player_ball_assigner=playerBallAssigner()
  team_ball_control=[]

  for frame_num,player_track in enumerate(tracks['player']):
     ball_bbox=tracks['ball'][frame_num][1]['bbox']
     assigned_player_id=player_ball_assigner.assign_ball_to_player(player_track,ball_bbox)

     if assigned_player_id is not None:
        tracks['player'][frame_num][assigned_player_id]['has_ball']=True
        team_ball_control.append(tracks['player'][frame_num][assigned_player_id]['team_color'])
     else:
        team_ball_control.append(team_ball_control[-1])  
     team_ball_control=np.array(team_ball_control)    
  output_frames=tracker.draw_annotations(video_frames,tracks,team_ball_control)
  save_video(output_frames,"output_video/output.mp4")
if __name__ == "__main__":
    main()    