from team_assigner.team_assigner import teamAssigner
from player_ball_assigner import playerBallAssigner
from utils import read_video, save_video
from tracker import Tracker
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
def main():
  # Read Video
  video_frames = read_video("input_video/input.mp4")

  # Initialize Tracker
  tracker = Tracker('model training/model/best.pt')

  # Get object tracks (Players, Referees, Ball)
  tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stub/tracks.pkl')

  # Estimate and adjust based on camera movement
  camera_movement_estimator = CameraMovementEstimator(video_frames[0])
  camera_movement_per_frame = CameraMovementEstimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stub/camera_movement.pkl')
  camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

  # Track objects' position and transform the view
  tracker.add_position_to_tracks(tracks)
  view_transformer = ViewTransformer()
  view_transformer.add_transformed_position_to_tracks(tracks)

  # Interpolate missing ball positions
  tracks['ball'] = tracker.interpolate_ball_position(tracks['ball'])

  # Estimate speed and distance of objects
  speed_and_distance_estimator = SpeedAndDistance_Estimator()
  speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

  # Assign players to their respective teams based on jersey colors
  team_assigner = teamAssigner()
  team_assigner.assign_team_color(video_frames[0], tracks['player'][0])

  for frame_num, player_track in enumerate(tracks['player']):
    for player_id, track in player_track.items():
        player_bbox = track['bbox']
        team_name = team_assigner.get_player_team(video_frames[frame_num], player_bbox, player_id)
        tracks['player'][frame_num][player_id]['team_color'] = team_assigner.team_color[team_name]
        
  # Assign the ball to the closest player and keep track of team ball control
  player_ball_assigner = playerBallAssigner()
  team_ball_control = []

  for frame_num, player_track in enumerate(tracks['player']):
     ball_bbox = tracks['ball'][frame_num][1]['bbox']
     assigned_player_id = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)

     if assigned_player_id is not None:
        tracks['player'][frame_num][assigned_player_id]['has_ball'] = True
        team_ball_control.append(tracks['player'][frame_num][assigned_player_id]['team_color'])
     else:
        team_ball_control.append(team_ball_control[-1])  
     team_ball_control = np.array(team_ball_control)    
     
  # Draw annotations on the video frames
  output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

  # Draw camera movement, speed, and distance indicators
  output_frames = CameraMovementEstimator.draw_camera_movement(output_frames, camera_movement_per_frame)
  speed_and_distance_estimator.draw_speed_and_distance(output_frames, tracks)

  # Save the resulting video
  save_video(output_frames, "output_video/output.avi")
if __name__ == "__main__":
    main()    