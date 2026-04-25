from sklearn.cluster import KMeans
class teamAssigner:
    """
    Class handling the assignment of players to teams based on the colors of their jerseys, using KMeans clustering.
    """
    def __init__(self):
        """
        Initializes the teamAssigner with empty dictionaries to store team colors and player team assignments.
        """
        self.team_color={}
        self.player_team={}
    
    def get_clustering_model(self,player_roi):
        """
        Fits a KMeans clustering model to an ROI (Region of Interest) of a player.

        Args:
            player_roi (numpy.ndarray): The image array representing the player.

        Returns:
            KMeans: A fitted scikit-learn KMeans model.
        """
        pixel_values=player_roi.reshape((-1,3))
        kmeans=KMeans(n_clusters=2,init='k-means++',n_init=1).fit(pixel_values)
        return kmeans
    
    def get_player_color(self,frame,bbox):
        """
        Extracts the dominant color (the jersey color) of a player from a bounding box.

        Args:
            frame (numpy.ndarray): The full video frame.
            bbox (list or tuple): The bounding box coordinates of the player [x1, y1, x2, y2].

        Returns:
            numpy.ndarray: The [R, G, B] or [B, G, R] color values for the player's jersey.
        """
        x1,y1,x2,y2=bbox
        player_roi=frame[int(y1):int(y2),int(x1):int(x2)]
        top_half_player=player_roi[0:player_roi.shape[0]//2,:]
        
        kmeans=self.get_clustering_model(top_half_player)
        labels=kmeans.labels_
        clustered_pixels=labels.reshape(top_half_player.shape[0],top_half_player.shape[1])
        
        corner_clusters=[clustered_pixels[0,0],clustered_pixels[0,-1],clustered_pixels[-1,0],clustered_pixels[-1,-1]]
        player_cluster=1-max(set(corner_clusters),key=corner_clusters.count)
        player_color=kmeans.cluster_centers_[player_cluster]
        return player_color
        
    def assign_team_color(self,frame,player_detections):
        """
        Clusters all players in an initial frame into two distinct team colors.

        Args:
            frame (numpy.ndarray): The video frame to extract colors from.
            player_detections (dict): Dictionary of player detections in the given frame.
        """
        player_colors=[]

        for _,player_detection in player_detections.items():
            bbox=player_detection['bbox']
            player_color=self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        cluster_palyer=KMeans(n_clusters=2,init='k-means++',n_init=1).fit(player_colors)    
        self.kmeans=cluster_palyer
        self.team_color['team_1']=cluster_palyer.cluster_centers_[0]
        self.team_color['team_2']=cluster_palyer.cluster_centers_[1]

    def get_player_team(self,frame,player_bbox,player_id): 
        """
        Predicts which team a given player belongs to, using the fitted KMeans team model.

        Args:
            frame (numpy.ndarray): The video frame.
            player_bbox (list or tuple): The player's bounding box [x1, y1, x2, y2].
            player_id (int): The tracker ID of the player.

        Returns:
            str: Team identifier ('team_1' or 'team_2').
        """
        if player_id in self.player_team:
            return self.player_team[player_id]
        player_color=self.get_player_color(frame,player_bbox)
        team_id=self.kmeans.predict(player_color.reshape(1,-1))[0]

        team_name='team_1' if team_id==0 else 'team_2'
        self.player_team[player_id]=team_name

        return team_name