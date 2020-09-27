import os, sys
import time
import datetime
import pandas as pd
import numpy as np
import math
from math import radians, cos, sin, asin, sqrt 
import random

from scipy.spatial import ConvexHull


ROOTDIR = os.path.abspath(os.path.realpath('./')) + '/Py'

sys.path.append(os.path.join(ROOTDIR, ''))

import dgckernel

import folium

import geopandas as gp


class Spatial_calculation(object):
    
    def __init__(self, Zoom):
        
        """ Load your trained model and initialize the parameters """
        self.Zoom=Zoom
        self.CALCULATOR = dgckernel.Calculator()
        self.CALCULATOR.SetLayer(Zoom)
        
    '''GRID ID'''

    def get_grid(self,lng,lat):

        return self.CALCULATOR.HexCellKey(dgckernel.GeoCoord(lat, lng))

    '''GRID SHAPE'''

    def get_grid_shape(self,grid):

        return self.CALCULATOR.HexCellVertexesAndCenter(grid)
        
    '''Neighbor Grid'''

    def grid_neighbor(self, grid, low_layer, up_layer):

        neighbors = self.CALCULATOR.HexCellNeighbor(grid, up_layer)
        _neighbors = self.CALCULATOR.HexCellNeighbor(grid, low_layer)
        neighbors = [e for e in neighbors if e not in _neighbors]
        return neighbors 
    
    def Geo_distance(self,lng1,lat1,lng2,lat2):
        lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) 
        dlon=lng2-lng1
        dlat=lat2-lat1
        a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 
        distance=2*asin(sqrt(a))*6371 
        distance=round(distance,2)
        return distance
    
    '''Get Distance'''
    def get_grid_dis(self,f_grid,t_grid):

        f_shape,f_center=self.get_grid_shape(f_grid);
        t_shape,t_center=self.get_grid_shape(t_grid);

        Topology_dis=1.3*self.Geo_distance(f_center.lng,f_center.lat,t_center.lng,t_center.lat)

        return Topology_dis
    
# Function that takes a map and a list of points (LON,LAT tupels) and
# returns a map with the convex hull polygon from the points as a new layer

def create_convexhull_polygon(map_object, list_of_points, layer_name, line_color, fill_color, weight, text): 

    # Since it is pointless to draw a convex hull polygon around less than 3 points check len of input
    if len(list_of_points) < 3:
        return

    # Create the convex hull using scipy.spatial 
    form = [list_of_points[i] for i in ConvexHull(list_of_points).vertices]

    # Create feature group, add the polygon and add the feature group to the map 
    fg = folium.FeatureGroup(name=layer_name)
    
    fg.add_child(folium.vector_layers.Polygon(locations=form, color=line_color, fill_color=fill_color,
                                              weight=weight, popup=(folium.Popup(text))))
    map_object.add_child(fg)

    return(map_object)

SC=Spatial_calculation(15)

def Convert_gps(POLYLINE):
    
    sample_trajectory=[[float(s) for s in seg.split(',')][::-1] for seg in str(POLYLINE).replace(']]','').replace('[','').split('],')]
    
    return sample_trajectory

def Convert_grid(Trajectory):
    
    sample_grid=[SC.get_grid(gps[1],gps[0]) for gps in Trajectory]
    
    return sample_grid

if __name__ == '__main__':

    df=pd.read_csv('data/train.csv')

    df=df.reset_index(drop=True)

    df=df[['TRIP_ID','POLYLINE']]

    df=df[df['POLYLINE']!='[]']

    df['Trajectory']=df.apply(lambda x:Convert_gps(x['POLYLINE']),axis=1)

    df['Grids']=df.apply(lambda x:Convert_grid(x['Trajectory']),axis=1)

    df.to_csv('data/train_df.csv')

