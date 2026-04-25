from utils import get_center_of_bbox,calc_distance

class playerBallAssigner:
    """
    Class responsible for assigning the ball to a specific player 
    based on proximity between the ball and players.
    """
    def __init__(self):
        """
        Initializes the playerBallAssigner with a maximum threshold distance 
        allowed for assigning the ball to a player.
        """
        self.max_player_ball_distance=70
        
    def assign_ball_to_player(self,players,ball_bbox):
        """
        Assigns the ball to the closest player within the maximum distance threshold.

        Args:
            players (dict): A dictionary of player tracks for a single frame, where keys are player IDs and values contain bounding box information.
            ball_bbox (list or tuple): Bounding box of the ball [x1, y1, x2, y2].

        Returns:
            int or None: The ID of the assigned player, or None if no player is close enough.
        """
        ball_position=get_center_of_bbox(ball_bbox)    
        mini_distance=float('inf')
        assigned_player_id=None
        for player_id,player in players.items():
            player_position=get_center_of_bbox(player['bbox'])
            distance_left=calc_distance((player_position[0],player_position[-1]),ball_position)
            distance_right=calc_distance((player_position[2],player_position[-1]),ball_position)
            distance=min(distance_left,distance_right)

            if distance<mini_distance and distance<self.max_player_ball_distance:
                mini_distance=distance
                assigned_player_id=player_id
        return assigned_player_id        