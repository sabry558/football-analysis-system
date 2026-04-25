import cv2

def read_video(path):
    """
    Reads a video file from the given path and returns a list of frames.

    Args:
        path (str): The path to the input video file.

    Returns:
        list: A list of numpy arrays, where each array is a frame of the video.
    """
    cap=cv2.VideoCapture(path)
    frames=[]
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames
def save_video(frames,path):
    """
    Saves a sequence of frames to a video file.

    Args:
        frames (list): A list of numpy arrays representing the video frames.
        path (str): The path where the output video will be saved.
    """
    height,width,_=frames[0].shape
    fourcc=cv2.VideoWriter_fourcc(*'XVID')
    out=cv2.VideoWriter(path,fourcc,24,(width,height))
    for frame in frames:
        out.write(frame)
    out.release() 