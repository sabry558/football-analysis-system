# Football Analysis System

An AI-powered computer vision project designed to analyze football (soccer) match videos. This system detects and tracks players, referees, and the ball using YOLO. It then applies several advanced analytical techniques such as team assignment based on jersey colors, player speed and distance calculations, court perspective transformation, camera movement adjustments, and ball possession tracking.

## Features

- **Object Tracking:** Uses a custom trained YOLO model to track players, referees, and the football.
- **Camera Movement Estimation:** Analyzes background optical flow to correct player and ball positions against the camera movement.
- **Perspective Transformation:** Transforms pixel coordinates from the 2D video feed into a top-down (birds-eye) perspective using field dimensions for real-world measurements.
- **Speed & Distance Estimation:** Calculates the real distances covered and running speeds of players across the field.
- **Team Assignment:** Uses KMeans clustering to identify and separate players into two distinguished teams based on their jersey colors.
- **Ball Possession Assignment:** Determines which team currently controls the football based on proximity logic.

## Project Architecture

- `main.py`: The entry point that ties all the analytical pipelines together.
- `tracker/`: YOLO and ByteTrack based object tracking module.
- `camera_movement_estimator/`: Estimates and offsets background motion.
- `view_transformer/`: Morphs object coordinates into top-down field coordinates.
- `speed_and_distance_estimator/`: Transforms pixel movements to real-world physics values (km/h & meters).
- `team_assigner/`: Clusters player regions into two distinct teams.
- `player_ball_assigner/`: Computes distances to determine ball possession.
- `utils/`: Common tools for video I/O and bounding box calculations (e.g., width, center, foot positions).

## Prerequisites & Installation

The project uses Python 3.11 and an assortment of machine learning and computer vision libraries. 

1. Ensure you have Python installed.
2. Activate your virtual environment (e.g., `.\myenv\Scripts\activate`).
3. Maintain dependencies (such as OpenCV, scikit-learn, Ultralytics YOLO, Supervision, Pandas, and Numpy).

## Usage

1. Place your input video inside the `input_video/` directory and ensure it is named `input.mp4`.
2. Ensure you have the trained YOLO weights (`best.pt`) located at `model training/model/best.pt`.
3. Run the main processing script:

```bash
python main.py
```

4. The processed, annotated video will be output to `output_video/output.avi` (or `output/` depending on the local configuration).

## Model Training
The `/model training/` directory contains the source Jupyter notebooks and Robo-flow YAML configurations used to originally train the YOLO model on football footages.