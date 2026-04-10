from utils import read_video, save_video
from tracker import Tracker
def main():
  video_frames=read_video("input.mp4")
  tracker=Tracker('model training/model/best.pt')
  tracks=tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path='stub/tracks.pkl')

if __name__ == "__main__":
    main()    