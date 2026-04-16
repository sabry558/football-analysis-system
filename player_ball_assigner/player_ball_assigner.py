from utils import get_center_of_bbox,calc_distance

class playerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance=70
    def assign_ball_to_player(self,players,ball_bbox):
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