from sklearn.cluster import KMeans
class teamAssigner:
    def __init__(self):
        self.team_color={}
        self.player_team={}
    
    def get_clustering_model(self,player_roi):
        pixel_values=player_roi.reshape((-1,3))
        kmeans=KMeans(n_clusters=2,init='k-means++',n_init=1).fit(pixel_values)
        return kmeans
    
    def get_player_color(self,frame,bbox):
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
        if player_id in self.player_team:
            return self.player_team[player_id]
        player_color=self.get_player_color(frame,player_bbox)
        team_id=self.kmeans.predict(player_color.reshape(1,-1))[0]

        team_name='team_1' if team_id==0 else 'team_2'
        self.player_team[player_id]=team_name

        return team_name