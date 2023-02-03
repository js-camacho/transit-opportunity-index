#--------------------------------------------------------------------
# Transport Opportunity Index (TOI) implementation version 2
#--------------------------------------------------------------------
'''
author: Johan Sebastian Camacho Diaz (@js.camacho)
'''
#------------------------
# Importing packages
#------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sys import getsizeof
import pickle
import math

from os import listdir, mkdir
from os.path import isfile, join
from collections import defaultdict
import geopandas as gp
import contextily as ctx
from geojson import Polygon
from shapely.geometry import Point, MultiPoint, LineString
from geopy.distance import geodesic
from datetime import datetime
from seaborn import heatmap
from sklearn import linear_model
from tqdm import tqdm

print('Packages imported!')

#--------------------------------
# Creating classes and functions
#--------------------------------
# TOI MODEL class

class ModelTOI():
    '''
    Description
    -----------
    
    Creates the shell for running TOI algorithms with differents sets of parameters and components.
    To initialize, is necessary to have a GTFS directory and a shapefile (e.g. GeoJSON) wiht the zones of analysis
    
    Parameters
    -----------
    
    * gtfs_path {str}
        Path to the GTFS directory
    
    * zones_path {str}
        Path to the file containing the geographical data of the zones of analysis
        
    * zone_id_co {False or str}
        The name of the column containing the indices for the zones of analysis within the shapefile.
    
    Methods
    -----------
    
    * get_direct_TOI()
        Runs the following methods to calculate the direct TOI:
            + filter_GTFS()
            + get_unique_trips()
            + get_travel_times()
            + create_stop_trip_shapes()
            + identify_representative_stops()
            + get_zone_travel_times()
            + calculate_spatial_coverage()
            + calculate_temporal_coverage(capacity, demand_col)
            + calculate_trip_coverage(alpha, beta, M, access_time, egress_time, wait_time)
            + calculate_direct_TOI()
        Then, outputs the direct TOI in three formats: TOI matrix, TOI origin and TOI destination
        For more detalis, read the documentation within each method.
        
    * get_indirect_TOI()
        After running get_direct_TOI(), runs the following methods to calculate the indirect TOI:
            + identify_transfer_stops()
            + get_indirect_zone_travel_times()
            + calculate_indirect_temporal_coverage()
            + calculate_indirect_trip_coverage()
            + calculate_indirect_TOI()
        Then, outputs the indirect TOI in three formats: TOI matrix, TOI origin and TOI destination
        For more detalis, read the documentation within each method.
        
    * plot_system()
        Creates visualizations of different stages of the algorithm.
        More details in the method documentation.
        
    * plot_TOI()
        Creates visualizations of the resulting TOI outputs.
        More details in the method documentation.
    
    Attributes
    -----------
    
    
    '''
    def __init__(self, gtfs_path, zones_path, zone_id_col=False):
        '''
        Initialization method
        '''
        self.gtfs = self.read_GTFS(gtfs_path)
        self.shapes = self.read_shapes(zones_path, zone_id_col)
    
    def read_GTFS(self, path):
        '''
        Description
        -----------

        Reads the GTFS files given a path. Returns them in a dictionary of DataFrames

        Parameters
        ----------

        * path {str}
            Path to a folder (not zip) contaning the GTFS files in csv format. These include, among others:
                - 'routes.csv'
                - 'stops.csv'
                - 'stop_times.csv'
                - 'trips.csv'
            All the files in the folder will be read as csv.

        Returns
        -------

        * gtfs {dict}
            A dictionary that represents a GTFS object.
            Each key is the name of a file in the GTFS: (stops, stop_times, routes, trips, etc.)
            Each value is a pandas.DataFrame object of the corresponding file.

        '''
        gtfs = {}
        files = [f for f in listdir(path) if isfile(join(path, f))]
        for f in files:
            gtfs[f[:-4]] = pd.read_csv(path+'/'+f, low_memory=False)
        return gtfs

    def read_shapes(self, path, zone_id_col=False):
        '''
        Description
        -----------

        Reads the GeoJSON file given a path or URL. Returns it as a GeoDataFrame

        Parameters
        ----------

        * path {str}
            Path or URL to a GeoJSON file

        * zone_id_col {col in shapes.columns}
            Name of the column containing the zone ids

        Returns
        -------

        * shapes {pandas.GeoDataFrame}
            A GeoDataFrame object containing the geographical data in Coordinate Reference System EPSG 4326
            A GeoDataFrame is simply a DataFrame with a "geometry" column that containg polygon objects with special methods and attributes

        '''
        shapes = gp.read_file(path, low_memory=False).to_crs(epsg='4326')
        shapes.dropna(inplace=True)
        if zone_id_col:
            shapes.set_index(zone_id_col, inplace=True)

        return shapes
    
    def get_direct_toi(self, time_window, days, capacity=1, max_walking_dist=1,
                      alpha=0.045507867152957786, beta=-0.04126873810720526, M=1, access_time=5, egress_time=5, wait_time='dynamic',
                      normalize=False):
        '''
        Description
        -----------
        
        Runs the following methods to calculate the direct TOI Supply:
            + filter_GTFS()
            + get_unique_trips()
            + get_travel_times()
            + create_stop_trip_shapes()
            + identify_representative_stops()
            + get_zone_travel_times()
            + calculate_spatial_coverage()
            + calculate_temporal_coverage(capacity, demand_col)
            + calculate_trip_coverage(alpha, beta, M, access_time, egress_time, wait_time)
            + calculate_direct_TOI()
        Then, outputs the direct TOI in three formats: TOI matrix, TOI origin and TOI destination
        For more details, read the documentation within each method
        
        Parameters
        ----------
        
        * time_window {(str,str)}
            The start_time and end_time to be considered in the analysis.
            E.g. time_window = ('5:00:00','8:00:00')
            All the trips considered should have a min(arrival_time) greater than start_time and a  max(arrival_time) less than end_time    
        
        * days {list}
            A list of the days of the week to be considered in the analysis.
            Only the trips for which the service_id operates for at least one of those days will be included. calendar DataFrame will be used for this.
        
        * capacity {True or int}
            True if there is a capacity.txt file in the GTFS with the capacities of each service_id.
            Otherwise, a constant value of capacity (int) should be specified.
        
        * max_walking_dist {float}
            Maximum distance in kilometers that users are willing to walk from their corresponding zone centroid to the nearest stop of a unique trip 
        
        * alpha {float}
            Decay function parameter (see decay_function() below)

        * beta {float}
            Decay function parameter (see decay_function() below)

        * M {float}
            Decay function parameter (see decay_function() below)

        * access_time {int}
            Average access time in minutes

        * egress_time {int}
            Average egress time in minutes

        * wait_time {int or 'dynamic'}
            Either an int representing the average wait time in minutes or a dictionary contaning the wait times by zone
        
        * normalize {bool}
          True to divide each value of TOI in the output by the total sum of TOI. This can aid measuring relative transit opportunity instead of absolute.
        
        Returns
        ----------
        
        * dir_toi_ij {pd.DataFrame}
            DataFrame containing the (origin, destination) values of the direct TOI. 
            Origin zones are in the rows and Destination zones are in the columns
            
        * dir_toi_i {pd.DataFrame}
            DataFrame containing the values of the direct TOI for each origin zone.
            This is equivalent to sum over all the columns of the TOI matrix
            
        * dir_toi_j {pd.DataFrame}
            DataFrame containing the values of the direct TOI for each destination zone.
            This is equivalent to sum over all the rows of the TOI matrix
        
        '''
        timer = datetime.now()
        print('-'*30+'\n'+'Computing Direct TOI...\n'+'-'*30)
        
        self.time_window = time_window
        self.filtered_gtfs = self.filter_GTFS(time_window, days)
        self.unique_trips = self.get_unique_trips(capacity)
        self.travel_times = self.get_travel_times()
        self.stops_shape, self.trips_shape, self.stop_trips_shape = self.create_stop_trip_shapes()
        self.repr_stops = self.identify_representative_stops(max_walking_dist)
        self.zone_travel_times = self.get_zone_travel_times()
        
        self.spatial_coverage = self.calculate_spatial_coverage(max_walking_dist)
        self.temporal_coverage = self.calculate_temporal_coverage(capacity, demand_col=False)
        if wait_time == 'dynamic':
            wait_times = self.get_wait_times(time_window)
        self.trip_coverage = self.calculate_trip_coverage(alpha, beta, M, access_time, egress_time, wait_time)
        
        self.dir_toi_ij, self.dir_toi_i, self.dir_toi_j = self.calculate_direct_TOI(normalize)
        print('-'*30+'\nTotal time elapsed:',datetime.now()-timer)
        return self.dir_toi_ij, self.dir_toi_i, self.dir_toi_j
    
    def get_indirect_toi(self, use_short_paths=True, max_transfer_dist=0.8, transfer_time=10, use_DAG=False, distances=False, capacity=1,
                        alpha=0.045507867152957786, beta=-0.04126873810720526, M=1, access_time=5, egress_time=5, wait_time='dynamic',
                        normalize=False, print_log=False):
        '''
        Description
        -----------

        After running get_direct_TOI(), runs the following methods to calculate the indirect TOI Supply:
            + identify_transfer_stops()
            + get_indirect_zone_travel_times()
            + calculate_indirect_temporal_coverage()
            + calculate_indirect_trip_coverage()
            + calculate_indirect_TOI()
        Then, outputs the indirect TOI in three formats: TOI matrix, TOI origin and TOI destination
        For more detalis, read the documentation within each method.
        
        Parameters
        ----------
        
        * use_short_paths {bool}
             If True, use a Shortest Path algorithm to find at most one indirect travel for each pair of zones.
             Read more in the documentation of get_shortest_indirect_travel_times() method.
        
        * max_transfer_dist {float}
            Maximum distance in kilometers that users are willing to walk from the stop of one route to a stop in a different trip.
            This distance between stops is computed using geodesic distance.
        
        * transfer_time: int or float or function
            If use_short_paths is False, then use a number representing the average transfer time in minutes.
            If use short_paths is True, use either a number representing a constant time to penalize transfers, or a function of the distance 
            (e.g. assume a constant velocity, then compute time).
        
        * distances {dict}
            Dictionary with (stop_i, stop_j) pairs as keys and their corresponding geodesic distances as values. Likely, output from precompute_distances()
            Providing a this distances matrix can drastically reduce the computational time for this step. (~1200 times faster for 4000 distinct stops)
            The function handles matrices that only record (i,j) values and not (j,i). Nevertheless, is necessary to have values for (i,i).
        
        * capacity {True or int}
            True if there is a capacity.txt file in the GTFS with the capacities of each service_id.
            Otherwise, a constant value of capacity (int) should be specified.
        
        * alpha {float}
            Decay function parameter (see decay_function() below)

        * beta {float}
            Decay function parameter (see decay_function() below)

        * M {float}
            Decay function parameter (see decay_function() below)

        * access_time {int}
            Average access time in minutes

        * egress_time {int}
            Average egress time in minutes

        * wait_time {int or dict}
            Either an int representing the average wait time in minutes or a dictionary contaning the wait times by zone
        
        * normalize {bool}
          True to divide each value of TOI in the output by the total sum of TOI. This can aid measuring relative transit opportunity instead of absolute.
        
        Returns
        ----------
        
        * indir_toi_ij {pd.DataFrame}
            DataFrame containing the (origin, destination) values of the indirect TOI. 
            Origin zones are in the rows and Destination zones are in the columns
            
        * indir_toi_i {pd.DataFrame}
            DataFrame containing the values of the indirect TOI for each origin zone.
            This is equivalent to sum over all the columns of the TOI matrix
            
        * indir_toi_j {pd.DataFrame}
            DataFrame containing the values of the indirect TOI for each destination zone.
            This is equivalent to sum over all the rows of the TOI matrix
        
        '''
        timer = datetime.now()
        print('-'*30+'\n'+'Computing Indirect TOI...\n'+'-'*30)
        
        self.use_short_paths = use_short_paths
        if use_short_paths:
            self.indirect_zone_travel_times = self.get_shortest_indirect_zone_travel_times(max_transfer_dist, transfer_time, distances, use_DAG)
        else:    
            self.transfer_stops = self.identify_transfer_stops(max_transfer_dist, distances, print_log)
            self.indirect_zone_travel_times = self.get_indirect_zone_travel_times(transfer_time, print_log)
        
        self.indirect_temporal_coverage = self.calculate_indirect_temporal_coverage(capacity, demand_col=False)
        if wait_time == 'dynamic':
            wait_time = self.get_wait_times(self.time_window)
        self.indirect_trip_coverage = self.calculate_indirect_trip_coverage(alpha, beta, M, access_time, egress_time, wait_time)
        
        self.indir_toi_ij, self.indir_toi_i, self.indir_toi_j = self.calculate_indirect_TOI(normalize)
        print('-'*30+'\nTotal time elapsed:',datetime.now()-timer)
        return self.indir_toi_ij, self.indir_toi_i, self.indir_toi_j
    
    def get_total_toi(self):
        '''
        Description
        -----------
        
        Combine the Direct and Indirect TOIs to get the Total TOI Supply values.
        It is necessary to run get_direct_toi() and get_indirect_toi() prior to this method.
        
        Returns
        ----------
        
        * toi_ij {pd.DataFrame}
            DataFrame containing the (origin, destination) values of the total TOI. 
            Origin zones are in the rows and Destination zones are in the columns
            
        * toi_i {pd.DataFrame}
            DataFrame containing the values of the total TOI for each origin zone.
            This is equivalent to sum over all the columns of the TOI matrix
            
        * toi_j {pd.DataFrame}
            DataFrame containing the values of the total TOI for each destination zone.
            This is equivalent to sum over all the rows of the TOI matrix
        '''
        self.toi_ij = self.dir_toi_ij + self.indir_toi_ij
        self.toi_i = self.dir_toi_i + self.indir_toi_i
        self.toi_j = self.dir_toi_j + self.indir_toi_j
        
        return self.toi_ij, self.toi_i, self.toi_j
    
    
    def get_toi_relative(self, demand_col):
        '''
        Description
        -----------

        After running get_direct_TOI(), get_indirect_TOI() and get_total_TOI() for the TOI Supply case (i.e. no demand proxy), recalculates all the TOI
        values by dividing them by their corresponding zone demands.
        If a the demand value for a zone is 0, the TOI value for this zone will be set to 0 by default.
        
        Parameters
        ----------
        
        * demand_col {False or column in shapes.columns}
            Name of the column in the <shapes> GeoDataFrame that contains the demand proxy.
        
        Returns
        ----------
        
        * dir_toi_ij_r {pd.DataFrame}
            Equivalent to dir_toi_ij but adjusted with the demand proxy
            
        * dir_toi_i_r {pd.DataFrame}
            Equivalent to dir_toi_i but adjusted with the demand proxy
            
        * dir_toi_j_r {pd.DataFrame}
            Equivalent to dir_toi_j but adjusted with the demand proxy
        
        * indir_toi_ij_r {pd.DataFrame}
            Equivalent to indir_toi_ij but adjusted with the demand proxy
            
        * indir_toi_i_r {pd.DataFrame}
            Equivalent to indir_toi_i but adjusted with the demand proxy
            
        * indir_toi_j_r {pd.DataFrame}
            Equivalent to indir_toi_j but adjusted with the demand proxy
            
        * toi_ij_r {pd.DataFrame}
            Equivalent to toi_ij but adjusted with the demand proxy
            
        * toi_i_r {pd.DataFrame}
            Equivalent to toi_i but adjusted with the demand proxy
            
        * toi_j_r {pd.DataFrame}
            Equivalent to toi_j but adjusted with the demand proxy
        '''
        f = lambda x: x if x != np.inf and not pd.isna(x) else 0
        
        self.dir_toi_ij_r = self.dir_toi_ij.divide(self.shapes[demand_col], axis=0).applymap(f)
        self.dir_toi_i_r = self.dir_toi_ij_r.sum(axis=1)
        self.dir_toi_j_r = self.dir_toi_ij_r.sum(axis=0)
        
        self.indir_toi_ij_r = self.indir_toi_ij.divide(self.shapes[demand_col], axis=0).applymap(f)
        self.indir_toi_i_r = self.indir_toi_ij_r.sum(axis=1)
        self.indir_toi_j_r = self.indir_toi_ij_r.sum(axis=0)
        
        self.toi_ij_r = self.toi_ij.divide(self.shapes[demand_col], axis=0).applymap(f)
        self.toi_i_r = self.toi_ij_r.sum(axis=1)
        self.toi_j_r = self.toi_ij_r.sum(axis=0)
        
        toi_list = [self.dir_toi_ij_r, self.dir_toi_i_r, self.dir_toi_j_r,\
                    self.indir_toi_ij_r, self.indir_toi_i_r, self.indir_toi_j_r,\
                    self.toi_ij_r, self.toi_i_r, self.toi_j_r]

        for t in toi_list:
            t.name = 'TOI'
            try:
                t.index = t.index.astype(int)  # For some reason it is set to type object by default for toi_j
            except:
                pass
        
        return toi_list
        
    
    def filter_GTFS(self, time_window=('00:00:01','23:59:59'), days=['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
        '''
        Description
        -----------

        Filter a GTFS object to include only the trips that are within a given <time_window> and that operate on specific <days>.
        To this end, the stop_times and trips DataFrames of the <gtfs> object will be modified and returned in a new <filtered_gtfs> dictionary.

        Parameters
        ----------
        
        * time_window {(str,str)}
            The start_time and end_time to be considered in the analysis.
            E.g. time_window = ('5:00:00','8:00:00')
            All the trips considered should have a min(arrival_time) greater than start_time and a  max(arrival_time) less than end_time

        * days {list}
            A list of the days of the week to be considered in the analysis.
            Only the trips for which the service_id operates for at least one of those days will be included. calendar DataFrame will be used for this.

        Returns
        -------

        * filtered_gtfs {dict}
            A dictionary that represents a GTFS object, with the stop_times and trips DataFrames filtered according to <time_window> and <days>

        '''
        trips = self.gtfs['trips'].copy()
        stop_times = self.gtfs['stop_times'].copy()
        calendar = self.gtfs['calendar'].copy()

        # Filtering by days
        timer = datetime.now()
        condition = calendar[days[0]] > 0
        for day in days:
            ## Creating a condition to filter trip_id for which the service_id is active on any of the selected days
            condition = condition | calendar[day] > 0
        active_services = list(calendar[condition].service_id)

        filtered_trips = trips[trips.service_id.apply(lambda x: x in active_services)]
        filtered_stop_times = stop_times.merge(filtered_trips[['trip_id']], how='inner', on='trip_id').sort_values(['trip_id','stop_sequence']).reset_index(drop=True)

        # Filtering by time window
        timer = datetime.now()
        start_time, end_time = time_window
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

        # Correcting arrival and departure times in case there are timestamps higher than '23:59:59'
        try:
            error = pd.to_datetime(filtered_stop_times.arrival_time.max()) > pd.to_datetime('23:59:00')
        except:
            error = filtered_stop_times.arrival_time.max() > '23:59:00'
        if error:
            # Substract 24 hours to the timestamps higher than 23:59:59 (e.g. 25:30:00 will be 01:30:00)
            filtered_stop_times['arrival_time'] = filtered_stop_times.arrival_time.apply(lambda x: '0'+str(int(x[:2])-24) + x[2:] if x > '23:59:59' else x)
            filtered_stop_times['departure_time'] = filtered_stop_times.departure_time.apply(lambda x: '0'+str(int(x[:2])-24) + x[2:] if x > '23:59:59' else x)

        filtered_stop_times.loc[:,('arrival_time')] = pd.to_datetime(filtered_stop_times.arrival_time)
        filtered_stop_times.loc[:,('departure_time')] = pd.to_datetime(filtered_stop_times.departure_time)

        # Grouping to get the min and max arrival time for each trip_id
        agg = filtered_stop_times.groupby('trip_id').arrival_time.agg(['min','max'])
        # Filtering trip_ids for which the min and max arrival times are within the time window
        active_trip_ids = agg[(agg['min'] >= start_time) & (agg['max'] <= end_time)].index.values

        filtered_trips = filtered_trips[filtered_trips.trip_id.apply(lambda x: x in active_trip_ids)]
        filtered_stop_times = stop_times.merge(filtered_trips[['trip_id']], how='inner', on='trip_id').sort_values(['trip_id','stop_sequence']).reset_index(drop=True)
        filtered_stop_times.stop_id = filtered_stop_times.stop_id.apply(lambda x: str(x))  # Casting all stop_ids to strings

        # Filtering stops to have only the stop_ids in filtered_stop_times
        stop_ids = filtered_stop_times[['stop_id']].drop_duplicates()
        stops = self.gtfs['stops'].copy()
        stops.stop_id = stops.stop_id.apply(lambda x: str(x))  # Casting all stop_ids to strings
        filtered_stops = stops.merge(stop_ids, how='inner', on='stop_id').sort_values('stop_id')

        # Filtering routes to have only the route_ids in filtered_trips
        route_ids = filtered_trips[['route_id']].drop_duplicates()
        routes = self.gtfs['routes'].copy()
        routes.route_id = routes.route_id.apply(lambda x: str(x))  # Casting all route_ids to strings
        filtered_routes = routes.merge(route_ids, how='inner', on='route_id').sort_values('route_id')

        # Creating new GTFS object to replace trips and stop_times
        filtered_gtfs = self.gtfs.copy()
        filtered_gtfs['trips'] = filtered_trips.reset_index(drop=True)
        filtered_gtfs['stop_times'] = filtered_stop_times.reset_index(drop=True)
        filtered_gtfs['stops'] = filtered_stops.reset_index(drop=True)
        filtered_gtfs['routes'] = filtered_routes.reset_index(drop=True)

        print('Filtered GTFS')
        print('  Time elapsed:',datetime.now()-timer)

        return filtered_gtfs

    
    def get_unique_trips(self, use_capacity=True):
        '''
        Description
        -----------

        Identify all the unique trips, that is, all the unique stop sequences found in the stop_times DataFrame within a GTFS object.
        For each unique trips, the frequency is computed, that is, how many times was each unique trip found in stop_times.
        Optionally, for each unique trip, the average capacity is also computed, based on the route_type of the unique trips.


        Parameters
        ----------
        
        * use_capacity {bool}
            True if there is a capacity.txt file in the GTFS with the capacities of each service_id

        Returns
        -------

        * unique_trips {dict}
            A dictionary containing all the unique trips in the keys and their corresponding [frequency, capacity] in the values.

        '''
        timer = datetime.now()
        stop_times = self.filtered_gtfs['stop_times'].copy()
        stop_times.sort_values(['trip_id','stop_sequence'], inplace=True)

        trips = self.filtered_gtfs['trips'].copy()
        routes = self.filtered_gtfs['routes'].copy()
        if use_capacity == True and type(use_capacity) == bool:
            capacity = self.filtered_gtfs['capacity'].copy()
            merged = pd.merge(pd.merge(trips,routes),capacity) # For each trip_id, there is an associated capacity, based on route_type
            capacities = {}

        else:
            merged = pd.merge(trips,routes)

        frequencies = {}
        for trip in stop_times.trip_id.unique():
            ## For each trip_id identify its corresponding stops sequence 
            stop_sequence = tuple(stop_times[stop_times.trip_id == trip].stop_id.values)
            if stop_sequence in frequencies.keys():
                frequencies[stop_sequence] += 1  ### If the sequence was recorded, add one to frequency
                if use_capacity == True and type(use_capacity) == bool:
                    capacities[stop_sequence].append(int(merged[merged.trip_id==trip].capacity))
            else:
                frequencies[stop_sequence] = 1   ### If the sequence is new, record it with frequecy one
                if use_capacity == True and type(use_capacity) == bool:
                    capacities[stop_sequence] = [int(merged[merged.trip_id==trip].capacity)]

        # Converting capacities list to average capacities of each unique trip
        if use_capacity == True and type(use_capacity) == bool:
            capacities = {key: np.mean(value) for key, value in capacities.items()}
            # Merging frequencies and capacities into a single dictionary
            unique_trips = {seq: [frequencies[seq], capacities[seq]] for seq in frequencies.keys()}

        else:
            unique_trips = {seq: frequencies[seq] for seq in frequencies.keys()}
        
        
        print('Unique trips identified:',len(unique_trips))
        print('  Time elapsed:',datetime.now()-timer)
        return unique_trips
    
    
    def get_travel_times(self, average=True):
        '''
        Description
        -----------

        Computes the in-vehicle travel times between all pairs of adjacent stops.
        Either a list of the travel times or the average of them can be returned.

        Parameters
        ----------

        * average {bool}
            If true, return the average travel times for each pair of adjacent stops.
            Otherwise, return the list of travel times for each pair of adjacent stops.

        Returns
        -------

        * travel_times {dict}
            A dictionary containing all pairs of adjacent stops in the keys and their corresponding travel times in the values.

        '''
        timer = datetime.now()
        stop_times = self.filtered_gtfs['stop_times'].copy()
        stop_times.sort_values(['trip_id','stop_sequence'], inplace=True)

        travel_times = {}
        prev_trip = stop_times.trip_id[0]
        prev_stop = stop_times.stop_id[0]
        prev_departure = stop_times.departure_time[0]

        # Starting from the second row, each row will be compared to the previous one
        for index, row in stop_times.iloc[1:,:].iterrows():
            ## If the trip_id is the same, then the stops are adjacent
            if row.trip_id == prev_trip:
                ### Fetching the adjacent stops pair
                stop_pair = (prev_stop, row.stop_id)
                ### Calculating time difference in minutes
                travel_time = (pd.to_datetime(row.arrival_time) - pd.to_datetime(prev_departure)).seconds/60
                if stop_pair in travel_times.keys():
                    travel_times[stop_pair].append(travel_time)  #### If the pair of stops was recorded, append the travel time to the list
                else:
                    travel_times[stop_pair] = [travel_time]      #### If the pair of stops is new, create a list with the travel time
            else:
                prev_trip = row.trip_id
            prev_stop = row.stop_id
            prev_departure = row.departure_time

        if average:
            ## Converting list of travel times to average travel times
            travel_times = {key:np.mean(value) for key, value in travel_times.items()}

        print('Travel times computed:',len(travel_times))
        print('  Time elapsed:',datetime.now()-timer)
        return travel_times
    
    
    def create_stop_trip_shapes(self):
        '''
        Description
        -----------

        Creates dicts of stops, trips and (stop,trips) pairs that will contain geometric shapely objects. These will be used in next steps during TOI calculation

        Returns
        -------

        * stops_shape {dict}
            Dictionary with stop_id in the keys and shapely.Point objects in the values

        * trips_shape {dict}
            Dictionary with unique trips in the keys and shapely.MultiPoints objects in the values, corresponding to the stops in that unique trip

        * stops_trips_shape {dict}
            Dictionary with (stop_id, unique trip) combinations in the keys and shapely.Point objects in the values, corresponding to the stop.
            One stop can belong to multiple unique trips.

        '''
        stops = self.filtered_gtfs['stops'].copy()

        stops_shape = {row['stop_id']: Point(row['stop_lon'],row['stop_lat']) for index, row in stops.iterrows()}
        trips_shape = {trip: MultiPoint([stops_shape[stop] for stop in trip]) for trip in self.unique_trips.keys()}
        stop_trips_shape = {(stop,trip): stops_shape[stop] for trip in self.unique_trips.keys() for stop in trip}

        return stops_shape, trips_shape, stop_trips_shape
    
    
    def identify_representative_stops(self, max_walking_dist=1, print_log=False):
        '''
        Description
        -----------

        Computes the representative stops.
        There is at most one representative stop for each (zone, unique_trip) pair.
        A representative stop is the one nearest to the centroid of a zone that belongs to a unique_trip and is closer than the <max_walking_dist>.
        They provide a way of connecting the unique trips to the zones. This will allow to get the travel times between zones by using the travel times between stops.

        Assumptions: 
            - Only stops that are closer to <max_walking_dist> in geodesic distance are reachable to users in a specific zone
            - Only the stop nearest to the centroid for a specific unique trip is considered, despite of being other stops with distances less than <max_walking_dist>

        Parameters
        ----------

        * max_walking_dist {float}
            Maximum distance in kilometers that users are willing to walk from their corresponding zone centroid to the nearest stop of a unique trip 

        Returns
        -------

        * repr_stops {dict}
            A dictionary in which:
                + Keys are all (zone, unique trip) pairs
                + Values are their corresponding [repr stop, distance between repr stop and centroid of the zone]
        '''
        timer = datetime.now()
        stops_ = self.filtered_gtfs['stops'].set_index('stop_id')
        repr_stops = {}
        i = 0

        for index, row in self.shapes.iterrows():
            ## There will be a representative stop for each (zone, unique_trip) pair
            zone_id = index
            zone_shape = row['geometry']

            if print_log:
                i += 1
                print('Zone',i,'done. Time elapsed:',datetime.now()-timer)

            for trip in self.unique_trips.keys():
                ### The representative stop will be the one closest to the centroid of the zone
                repr_stop = None
                min_dist = 1e9
                for stop in trip:
                    #### Computing the distance between a stop and the centroid of a zone, using geodesic distance
                    zone_lon, zone_lat = zone_shape.centroid.xy
                    zone_lon, zone_lat = zone_lon[0], zone_lat[0]
                    dist = geodesic((zone_lat,zone_lon),(stops_.loc[stop,['stop_lat','stop_lon']])).km
                    #### Alternatively, using Euclidean distance
                    #dist = zone_shape.centroid.distance(stops_shape[stop])
                    if dist < min_dist:
                        min_dist = dist
                        repr_stop = stop
                if min_dist < max_walking_dist:
                    repr_stops[(zone_id,trip)] = [repr_stop,min_dist]
                else:
                    repr_stops[(zone_id,trip)] = [False,False]
        
        
        print('Representative stops identified:',len(repr_stops))
        print('  Time elapsed:',datetime.now()-timer)
        return repr_stops
    
    
    def get_zone_travel_times(self):
        '''
        Description
        -----------

        Compute average travel times between zones (trip coverage before decay function) using representative stops
        To compute the travel time for a triplet (zone_i, zone_j, unique_trip) do the following:
            1. For each (zone_i, zone_j, unique_trip) check if there is a representative stop for (zone_j, unique_trip) and (zone_i, unique_trip)
            2. If there is, it means that the zones are connected by that unique_trip
                2.1. Find the adjacent stops connecting them and add their travel times (it could be the same stop, which will be a total travel time of 0)
                    2.1.1. There is an exception in which a unique_trip is not bidirectional, so in the case of opposite direction binary connectivity factor is 0
                2.2. Set the binary connectivity factor to 1
            3. If not, the two zones are not connected by that unique_trip, so set the binary connectivity factor to 0

        Returns
        -------

        * zone_travel_times {dict}
            Dictionary where:
                - Keys: All triplets (zone_i, zone_j, unique_trip) that are directly connected
                - Values: Average in-vehicle travel times between the repr stops from zone_i and zone_j using adajcent stops in unique_trip

        '''
        timer = datetime.now()  # Timer
        zone_travel_times = {}  # Travel times for each triplet (zone_i, zone_j, unique_trip)

        zones = self.shapes.index.to_list()
        for zone_i in zones:
            zones2 = zones.copy()
            zones2.remove(zone_i)
            for zone_j in zones2:
                for trip in self.unique_trips.keys():
                    stop_i = self.repr_stops[(zone_i,trip)][0]
                    stop_j = self.repr_stops[(zone_j,trip)][0]
                    #### Checking that there is a representative stop for both zones in a specific unique trip
                    if stop_i and stop_j:  
                        ##### Handling exception in case the representative stop for both zones is the same
                        if stop_i == stop_j:
                            zone_travel_times[(zone_i, zone_j, trip)] = 0
                        else:
                            stops_subset = []  # This list will store the subset of stops that connect stop_i and stop_j within the unique trip
                            start = False
                            end = False
                            ###### Getting travel time between two stops by adding the travel times of adjacent stops
                            for stop in trip:
                                if stop == stop_i:
                                    stops_subset.append(stop)
                                    start = True
                                elif stop == stop_j and start:
                                    stops_subset.append(stop)
                                    end = True
                                    break  # Breaking loop in case the start stop happens more than once in the trip
                                elif start and not end:
                                    stops_subset.append(stop)
                            ###### Handling exception in case both zones are covered by the same unique trip, but the trip goes in opposite direction only
                            if not end:
                                pass
                            ##### Otherwise, the travel times between adjacent stops are added up to ge the travel time between zones
                            else:
                                total_travel_time = 0
                                for s1, s2 in zip(stops_subset[:-1], stops_subset[1:]):  # Creating adjacent stop pairs
                                    total_travel_time += self.travel_times[s1,s2]
                                zone_travel_times[(zone_i, zone_j, trip)] = total_travel_time
                    #### If there is no representative stop for each zone in a unique trip, the binary connectivity factor is 0    
                    pass
        
        
        print('Direct zone travel times calculated:', len(zone_travel_times))
        print('  Time elapsed:',datetime.now()-timer)
        return zone_travel_times
    
    
    def calculate_spatial_coverage(self, max_walking_dist=1):
        '''
        Description
        -----------

        Calculate the spatial coverage for each (zone, unique trip) pair.
        Spatial coverage is the proportion of a zone area covered by the stops of a particular trip, after a euclidean buffer is applied 
        to each stop in the unique trip.
        The buffer is a circular area around a stop with radius <max_walking_dist> in kilometers.

        Assumptions:
            - The service (number of adjusted seats) offered is proportional to the area covered by the buffers

        Parameters
        ----------

        * max_walking_dist {float}
            Maximum distance in kilometers that users are willing to walk from their corresponding zone centroid to the nearest stop of a unique trip

        Returns
        -------

        * spatial_coverage {dict}
            A dictionary with (zone, unique trip) pairs in the keys and their corresponding spatial coverages in the values

        '''
        timer = datetime.now()
        spatial_coverage = {}

        buffer = 0.009*max_walking_dist   # 0.009 units in lat_lon euclidean distance is approximately 1 km in geodesic distance. Check geodesic((0,0),(0,0.009)).km

        for zone_id, row in self.shapes.iterrows():
            zone_shape = row['geometry']
            ## There will be a spatial coverage for each (zone, unique trip) pair
            for trip, trip_shape in self.trips_shape.items():
                ### Creating the Multipoint buffer geometry
                trip_buffer = trip_shape.buffer(buffer)
                ### Computing the area of the intersection over the total area of the zone
                area_covered = trip_buffer.intersection(zone_shape).area/zone_shape.area
                spatial_coverage[(zone_id, trip)] = area_covered
        
        print(f'Spatial Coverage calculated. {len(spatial_coverage)} (zone, unique_trip) pairs identified')
        print('  Time elapsed:',datetime.now()-timer)
        return spatial_coverage
    
    
    def calculate_temporal_coverage(self, capacity=True, demand_col=False):
        '''
        Description
        -----------

        Calculates the temporal coverage for each (zone_i, zone_j, unique trip) triplet.
        The temporal coverage calculated as the frequency of the buses using the unique trip multiplied by the capacity of each one.
        Thus, the unit of measure is number of seats offered to traverse from zone_i to zone_j using a specific unique trip.
        Optionally, this value can be divided by a demand proxy of zone_i. In that case, the unit of measure would be number of seats offered per customer.

        Note that this values will also be multiplied by the binary connectivity parameter in <connected> so that only directly connected zones have
        temporal coverage greater than 0.

        Assumptions:
            - The frequency and capacity only depend on the unique trip, not on the pair of zones

        Parameters
        ----------

        * capacity {True or int}
            If True, assumes the unique_trips values are tuples (frequency, capacity).
            Otherwise, an int should be passed as the constant capacity for all trips.

        * demand_col {False or column in shapes.columns}
            If False, temporal coverage would not be divided by demand.
            Else, the name of the column in the <shapes> GeoDataFrame that contains the demand proxy must be given.

        Returns
        -------

        * temporal_coverage {dict}
            - Keys: All triplets of (zone_i, zone_j, unique trip)
            - Values: The corresponding temporal coverage as described above.

        '''
        timer = datetime.now()
        temporal_coverage = {}

        for (zone_i,zone_j,trip) in self.zone_travel_times.keys():
            # The frequency and capacity only depends on the unique trip, not on the pair of zones
            if capacity == True and type(capacity) == bool:
                freq, cap = self.unique_trips[trip]
            # If there is no capacity in unique_trips, use a constant value
            else:
                freq = self.unique_trips[trip]
                cap = capacity
            if demand_col:
                demand = self.shapes.loc[zone_i,demand_col]
            else:
                demand = 1

            if demand > 0:
                temporal_coverage[(zone_i,zone_j,trip)] = freq*cap/demand
            else:
                temporal_coverage[(zone_i,zone_j,trip)] = 0

        print(f'Temporal Coverage calculated. {len(temporal_coverage)} (zone_i, zone_j, unique_trip) tuples identified')
        print('  Time elapsed:',datetime.now()-timer)
        return temporal_coverage
    
    
    def calculate_trip_coverage(self, alpha=0.045507867152957786, beta=-0.04126873810720526, M=1, access_time=5, egress_time=5, wait_time=5):
        '''
        Description
        -----------

        Calculates the trip coverage for each (zone_i, zone_j, unique trip) triplet.
        The trip coverage is computed as the average travel time from zone_i to zone_j using a given unique trip
        after passing it through a decay function (receives travel time in minutes and outputs perception of service between 0 and 1).
        The unit of measure of the trip coverage is perception of service of connected zones according to the travel time using a specific unique trip.

        Assumptions:
            - The service (number of adjusted seats) offered is proportional to the perception of service according to the average travel time between zones

        Parameters
        ----------

        * alpha {float}
            Decay function parameter (see decay_function() below)

        * beta {float}
            Decay function parameter (see decay_function() below)

        * M {float}
            Decay function parameter (see decay_function() below)

        * access_time {int}
            Average access time in minutes

        * egress_time {int}
            Average egress time in minutes

        * wait_time {int or dict}
            Either an int representing the average wait time in minutes or a dictionary contaning the wait times by zone

        Returns
        -------

        * trip_coverage {dict}
            - Keys: All triplets of (zone_i, zone_j, unique trip)
            - Values: The corresponding trip coverage as described above.

        '''
        timer = datetime.now()
        trip_coverage = {}
        for (zone_i,zone_j,trip), T in self.zone_travel_times.items():
            # Constant wait_time
            if type(wait_time) == int:
                T += access_time + egress_time + wait_time
            # Wait_time depending on the zone frequency
            elif type(wait_time) == dict:
                T += access_time + egress_time + wait_time[zone_i]
            trip_coverage[(zone_i,zone_j,trip)] = self.decay_function(T, alpha, beta, M)

        print(f'Trip Coverage calculated. {len(trip_coverage)} (zone_i, zone_j, unique_trip) tuples identified')
        print('  Time elapsed:',datetime.now()-timer)
        return trip_coverage
    
    def decay_function(self, T, alpha=0.045507867152957786, beta=-0.04126873810720526, M=1):
        '''
        Computes the output of the Decay Function to estimate the service level using the given parameters
        '''
        f = M/(1+alpha*np.exp(-beta*T))
        return f
    
    def calculate_direct_TOI(self, normalize=False):
        '''
        Description
        -----------

        Calculates the Direct TOI from the original 2018 paper by combining spatial, temporal and trip coverage.
        These elements are combined using product and can be expressed in 3 different ways: origin, destination or origin-destination values.
        If demand proxy was not included in temporal coverage, the unit of measure of TOI is the adjusted number of seats offered by the transportation system.
        If demand proxy was included, the unit of measure of TOI is the adjusted number of seats per capita offered by the transportation system.

        Parameters
        ----------
        
        * normalize {bool}
          True to divide each value of TOI in the output by the total sum of TOI. This can aid measuring relative transit opportunity instead of absolute.

        Returns
        -------

        * TOI {pd.DataFrame}
            DataFrame containing the (origin, destination) values of the direct TOI. 
            Origin zones are in the rows and Destination zones are in the columns
            
        * TOI_i {pd.DataFrame}
            DataFrame containing the values of the direct TOI for each origin zone.
            This is equivalent to sum over all the columns of the TOI matrix
            
        * TOI_j {pd.DataFrame}
            DataFrame containing the values of the direct TOI for each destination zone.
            This is equivalent to sum over all the rows of the TOI matrix

        '''
        timer = datetime.now()
        TOI = pd.DataFrame(index=self.shapes.index, columns=self.shapes.index, dtype=float)
        TOI.index.name = 'Zone'

        for zone_i in self.shapes.index:
            for zone_j in self.shapes.index:
                if zone_i != zone_j:
                    ### Aggregate over all the unique trips
                    toi = 0
                    for trip in self.unique_trips:
                        spat_cov = self.spatial_coverage[zone_i,trip]
                        temp_cov = self.temporal_coverage[zone_i,zone_j,trip] if (zone_i,zone_j,trip) in self.temporal_coverage.keys() else 0
                        trip_cov = self.trip_coverage[zone_i,zone_j,trip] if (zone_i,zone_j,trip) in self.trip_coverage.keys() else 0  
                        toi += spat_cov*temp_cov*trip_cov
                    TOI.loc[zone_i,zone_j] = toi

        if normalize:
            total = TOI.sum().sum()
            for index, row in TOI.iterrows():
                for col in TOI.columns:
                    row[col] /= total

        TOI_i = TOI.sum(axis=1)
        TOI_i = TOI_i.astype(float)
        TOI_j = TOI.sum(axis=0)
        TOI_j = TOI_j.astype(float)

        TOI_i.name = 'TOI'
        TOI_j.name = 'TOI'
        TOI.name = 'TOI'

        print('Direct TOI computed. TOI shape:',TOI.shape)
        print('  Time elapsed:',datetime.now()-timer)

        return TOI, TOI_i, TOI_j
    
    
    def identify_transfer_stops(self, max_transfer_dist=0.8, distances=False, print_log=False):
        '''
        Description
        -----------

        Identify the pair of stops that could allow a transfer between each pair of unique trips, 
        under the condition that the distance between them is less than <max_transfer_dist> (in kilometers).

        For efficiency, this algorithm can use precomputed distances to avoid doing a lot of geodesic distance operations.
        If a specific distance between two stops was not given, then it is computed once and stored so that next time it is fetched instead of computed.
        User can either provide an empty distance dictionary, an incomplete distance dictionary or a complete distance dictionary.
        In any case, the dictionary will be filled for the missing distances and returned if user asks for it.

        Assumptions:
            - The distance between stops is approximated using geodesic distance, so it is symmetric for each pair of stops.
            - Because of the above assumption, the transfers are symmetric too. So only (i,j) is computed, avoiding (j,i)
            - There is no consideration of the arrival/departure times in the condition for creating a transfer
            - Only the pair of stops with the closest distance is candidate for a transfer, so there can only be one transfer option for each pair of trips.

        Parameters
        ----------

        * max_transfer_dist {float}
            Maximum distance in kilometers that users are willing to walk from the stop of one route to a stop in a different trip.
            This distance between stops is computed using geodesic distance.

        * distances {False or str path}
            Path to a pickle file (.pkl) containing a dictionary with (stop_i, stop_j) pairs as keys and their corresponding geodesic distances as values. 
            Providing a this distances matrix can drastically reduce the computational time for this step. (~1200 times faster for 4000 distinct stops)
            The function handles matrices that only record (i,j) values and not (j,i). Nevertheless, is necessary to have values for (i,i).

        Returns
        -------

        * transfer_stops {dict}
            A dictionary in which:
                - Keys: All possible distinct unique trip pairs (only (i,j) not (j,i))
                - Values: List of two elements if there is a transfer, False otherwise:
                    + Pair of transfer stops
                    + Geodesic distance in kilometers between these two stops
        '''
        timer = datetime.now()
        stops = self.filtered_gtfs['stops'].set_index('stop_id')

        # Symmetric case: compute only (trip_i,trip_j) pairs only, instead of (trip_i,trip_j) and (trip_j,trip_i)
        transfer_stops = {}
        tot_transf = 0
        
        # Reading distances if a path is given, else create empty dictionary
        if distances == False:
            distances = {}
        else:
            with open(distances, 'rb') as f:
                distances = pickle.load(f)

        i = -1
        for trip1 in self.unique_trips:
            i += 1
            if print_log:
                print('Trip',i,'. Time elpased',datetime.now()-timer)
            for trip2 in list(self.unique_trips.keys())[i:]:
                # For each pair of distinct trips, identify transfer stops
                if trip1 != trip2:
                    ## Compute the distance between all pairs of stops between the two trips
                    min_pair = None
                    min_dist = 1e9
                    for s1 in trip1:
                        for s2 in trip2:
                            if s1 == s2:
                                dist = 0
                            else:
                                ## Computes only distances on-demand and save them for future use
                                try:
                                    dist = distances[(s1,s2)]
                                ## Since distances may store only (i,j) values and not (j,i), we handle this excpetion
                                except:
                                    try:
                                        dist = distances[(s2,s1)]
                                    ### If distance has not been computed before, calculate it and then save it
                                    except:
                                        dist = self.geodesic2((stops.loc[s1,['stop_lat','stop_lon']]),(stops.loc[s2,['stop_lat','stop_lon']]))
                                        distances[(s1,s2)] = dist
                            if dist < min_dist:
                                min_dist = dist
                                min_pair = (s1,s2)
                    if min_dist < max_transfer_dist:
                        transfer_stops[(trip1,trip2)] = [min_pair,min_dist]
                        tot_transf += 1
                    #else:
                    #    transfer_stops[(trip1,trip2)] = [False,False]

        print(f'{tot_transf} Transfers identified.')
        print('  Time elapsed:',datetime.now()-timer)
        
        # Writing distances to a pickle file
        with open('stop_distances.pkl', 'wb') as f:
            pickle.dump(distances, f)
        
        return transfer_stops
    
    
    def get_indirect_zone_travel_times(self, transfer_time=10, print_log=False):
        '''
        Description
        -----------

        Compute the indirect travel times between pairs of zones according to the available transfers in <transfer_stops>.
        An indirect travel starts at one zone in a specific trip, then transfers to other trip to arrive at a different zone.

        To calculate indirect zone travel time we have the following conditions:
            ~ There must be a feasible transfer between the pair of trips
            ~ The departure trip (trip1) must have a representative stop in the origin zone (zone_i) 
                ~ This representative stop of zone_i must precede the start transfer stop (transfer_stop1) in the stop sequence of trip1
            ~ The arrival trip (trip2) must have a representative stop in the destination zone (zone_j)
                ~ This representative stop of zone_j must succeed the end transfer stop (transfer_stop2) in the stop sequence of trip2

        If these conditions are met, the calculation is as follows (add the following quantities):
            + In-vehicle travel time from repr_stop of trip1 in zone_i to transfer_stop1 (this must be in trip1)
            + Average transfer time (<transfer_time>)
            + In-vehicle travel time from transfer_stop2 (must be in trip2) to the repr_stop of trip2 in zone_j

        Assumptions:
            - There is only one possible transfer between each pair of routes
            - An indirect travel cannot have more than one transfer
            - We only consider the representative stops as the departure and arrival stops for the travel

        Parameters
        ----------

        * transfer_time {float or int}
            Average transfer time (in minutes) to be added to all indirect travel times
            This represent the time a user has to spend walking during the transfer

        Returns
        -------

        * indirect_zone_travel_times {dict}
            Dictionary where:
                - Keys: All tuples (zone_i, zone_j, trip1, trip2)
                - Values: Average travel times to go from zone_i to zone_j by using trip1 and trip2.

        '''
        timer = datetime.now()
        indirect_zone_travel_times = {} # Travel times for each tuple (zone_i, zone_j, unique_trip1, unique_trip2)

        zones = self.shapes.index.to_list()
        i = 0
        for zone_i in zones:
            if print_log:
                i += 1
                print('Zone',i,'done. Time:', datetime.now()-timer)
            for zone_j in zones:
                if zone_i != zone_j:
                    ## For each pair of zones, check which of all possible transfers are feasible for indirectly connecting the two zones
                    for (trip1, trip2), (transfer, dist) in self.transfer_stops.items():
                        if transfer:
                            transf_stop1, transf_stop2 = transfer
                        stop1 = self.repr_stops[(zone_i,trip1)][0]
                        stop2 = self.repr_stops[(zone_j,trip2)][0]
                        ### Checking that there is a representative stop for both zones in a specific unique trip and that the trasnfer between trips exist
                        if stop1 and stop2 and transfer:
                            ### Since unique trips are not bidirectional in general, we have the following precedence conditions for a feasible transfer
                            if (trip1.index(transf_stop1) > trip1.index(stop1)) and (trip2.index(transf_stop2) < trip2.index(stop2)):
                                stop_subset1 = trip1[trip1.index(stop1):trip1.index(transf_stop1)+1] # Selecting the stops in trip1 from stop1 to transf_stop1
                                stop_subset2 = trip2[trip2.index(transf_stop2):trip2.index(stop2)+1] # Selecting the stops in trip2 from trasnf_stop2 to stop2
                                #### Computing travel time as described in the function description
                                total_travel_time = transfer_time
                                for s1,s2 in zip(stop_subset1[:-1] + stop_subset2[:-1], stop_subset1[1:] + stop_subset2[1:]):
                                    total_travel_time += self.travel_times[s1,s2]
                                indirect_zone_travel_times[(zone_i, zone_j, trip1, trip2)] = total_travel_time
                        ### If there is no representative stop for each (zone, trip) pair, then the binary connectivity factor is 0
                        else:
                            pass

        print(f'Indirect zone travel times computed. {len(indirect_zone_travel_times)} (zone_i, zone_j, trip1, trip2) tuples found')
        print('  Time elapsed:',datetime.now()-timer)
        return indirect_zone_travel_times
    
    
    def calculate_indirect_temporal_coverage(self, capacity=True, demand_col=False):
        '''
        Description
        -----------

        Calculates the indirect temporal coverage for each (zone_i, zone_j, trip1, trip2) tuple.
        The temporal coverage calculated as the frequency of the buses using the unique trip multiplied by the capacity of each one.
        Thus the unit of measure is number of seats offered to traverse from zone_i to zone_j using a specific unique trip.
        Optionally, this value can be divided by a demand proxy of zone_i. In that case, the unit of measure would be number of seats offered per customer.

        Note that this values will also be multiplied by the binary indirect connectivity parameter in <connected> so that only directly connected zones have
        temporal coverage greater than 0.

        Assumptions:
            - The frequency and capacity only depend on the pair of trips, not on the pair of zones

        Parameters
        ----------

         * shapes {pandas.GeoDataFrame}
            A GeoDataFrame object containing the geospatial data of the zones in Coordinate Reference System EPSG 4326

        * unique_trips {dict}
            A dictionary containing all the unique trips in the keys and their corresponding frequencies in the values

        * indirect_zone_travel_times {dict}
            Dictionary where:
                - Keys: All tuples (zone_i, zone_j, trip1, trip2)
                - Values: Average travel times to go from zone_i to zone_j by using trip1 and trip2.

        * capacity {True or int}
            If True, assumes the unique_trips values are tuples (frequency, capacity).
            Otherwise, an int should be passed as the constant capacity for all trips.

        * demand_col {False or column in shapes.columns}
            If False, temporal coverage would not be divided by demand.
            Else, the name of the column in the <shapes> GeoDataFrame that contains the demand proxy must be given.

        Returns
        -------

        * indirect_temporal_coverage {dict}
            - Keys: All tuples of (zone_i, zone_j, trip1, trip2)
            - Values: The corresponding temporal coverage as described above.

        '''
        timer = datetime.now()
        indirect_temporal_coverage = {}
        for (zone_i,zone_j,trip1,trip2) in self.indirect_zone_travel_times.keys():
            # In the Indirect TOI, is specified that only lowest temporal coverage of the two legs (trip1 and trip2) is considered
            if capacity == True and type(capacity) == bool:
                freq1, cap1 = self.unique_trips[trip1]
                freq2, cap2 = self.unique_trips[trip2]
                freq_x_cap = min(freq1*cap1, freq2*cap2)
            # If there is no capacity in unique_trips, use a constant value
            else:
                freq1 = self.unique_trips[trip1]
                freq2 = self.unique_trips[trip2]
                freq_x_cap = min(freq1*capacity, freq2*capacity)
            if demand_col:
                demand = self.shapes.loc[zone_i,demand_col]
            else:
                demand = 1
                
            if demand > 0:
                indirect_temporal_coverage[(zone_i,zone_j,trip1,trip2)] = freq_x_cap/demand
            else:
                indirect_temporal_coverage[(zone_i,zone_j,trip1,trip2)] = 0

        print(f'Indirect Temporal Coverage calculated. {len(indirect_temporal_coverage)} (zone_i, zone_j, trip1, trip2) tuples found')
        print('  Time elapsed:',datetime.now()-timer)
        return indirect_temporal_coverage
    
    
    def calculate_indirect_trip_coverage(self, alpha=0.045507867152957786, beta=-0.04126873810720526, M=1, access_time=5, egress_time=5, wait_time=5):
        '''
        Description
        -----------

        Calculates the indirect trip coverage for each (zone_i, zone_j, trip1, trip2) tuple.
        The trip coverage is computed as the binary connectivity parameter multiplied by the average travel time from zone_i to zone_j doing a tranfer from
        trip1 to trip2 after passing through a decay function (receives travel time and outputs perception of service between 0 and 1).
        The unit of measure of the trip coverage is perception of service of connected zones according to the travel time using a specific unique trip.

        Parameters
        ----------

        * indirect_connected {dict}
            Dictionary where:
                - Keys: All tuples (zone_i, zone_j, trip1, trip2)
                - Values: Binary connectivity factor. 1, if the two zones are connected by using a transfer from trip1 to trip2. 0, otherwise.

        * alpha {float}
            Decay function parameter (see decay_function() below)

        * beta {float}
            Decay function parameter (see decay_function() below)

        * M {float}
            Decay function parameter (see decay_function() below)

        Returns
        -------

        * indirect_trip_coverage {dict}
            - Keys: All tuples (zone_i, zone_j, trip1, trip2)
            - Values: The corresponding trip coverage as described above.

        '''
        timer = datetime.now()
        indirect_trip_coverage = {}
        for (zone_i,zone_j,trip1,trip2), T in self.indirect_zone_travel_times.items():
            # Constant wait_time
            if type(wait_time) == int:
                T += access_time + egress_time + wait_time*2
            # Wait_time depending on the zone frequency
            elif type(wait_time) == dict:
                T += access_time + egress_time + wait_time[zone_i] + wait_time[zone_j]
            indirect_trip_coverage[(zone_i,zone_j,trip1,trip2)] = self.decay_function(T, alpha, beta, M)

        print(f'Indirect Trip Coverage calculated. {len(indirect_trip_coverage)} (zone_i, zone_j, trip1, trip2) tuples found')
        print('  Time elapsed:',datetime.now()-timer)
        return indirect_trip_coverage
    
    def calculate_indirect_TOI(self, normalize=False):
        '''
        Description
        -----------

        Calculates the Indirect TOI from the original 2018 paper by combining spatial, temporal and trip coverage.
        These elements are combined using product and can be expressed in 3 different ways: origin, destination or origin-destination values.
        If demand proxy was not included in temporal coverage, the unit of measure of TOI is the adjusted number of seats offered by the transportation system.
        If demand proxy was included, the unit of measure of TOI is the adjusted number of seats per capita offered by the transportation system.

        Main differences with the way that TOI is calculated are:
            - spatial_coverage is calcualted using only the first leg of the trip
            - temporal_coverage is calculated using the leg of the trip with the lowest service frequency
            - trip_coverage is only considered for travels with exactly one transfer

        Note that the output TOI is only considering transfer travels. This output can be easily added to the Direct TOI to get the complete view of the system.

        Parameters
        ----------
        
        * normalize {bool}
          True to divide each value of TOI in the output by the total sum of TOI. This can aid measuring relative transit opportunity instead of absolute.

        Returns
        -------

        * TOI {pd.DataFrame}
            DataFrame containing the (origin, destination) values of the indirect TOI. 
            Origin zones are in the rows and Destination zones are in the columns
            
        * TOI_i {pd.DataFrame}
            DataFrame containing the values of the indirect TOI for each origin zone.
            This is equivalent to sum over all the columns of the TOI matrix
            
        * TOI_j {pd.DataFrame}
            DataFrame containing the values of the indirect TOI for each destination zone.
            This is equivalent to sum over all the rows of the TOI matrix

        '''  
        timer = datetime.now()
        TOI = pd.DataFrame(data=0, index=self.shapes.index, columns=self.shapes.index, dtype=float)
        TOI.index.name = 'origin'
        TOI.columns.name = 'destination'

        for (zone_i,zone_j,trip_i,trip_j) in self.indirect_zone_travel_times:
            spat_cov = self.spatial_coverage[zone_i,trip_i]
            temp_cov = self.indirect_temporal_coverage[zone_i,zone_j,trip_i,trip_j]
            trip_cov = self.indirect_trip_coverage[zone_i,zone_j,trip_i,trip_j]
            TOI.loc[zone_i,zone_j] += spat_cov*temp_cov*trip_cov
        

        if normalize:
            total = TOI.sum().sum()
            for index, row in TOI.iterrows():
                for col in TOI.columns:
                    row[col] /= total

        TOI_i = TOI.sum(axis=1)
        TOI_i = TOI_i.astype(float)
        TOI_j = TOI.sum(axis=0)
        TOI_j = TOI_j.astype(float)

        TOI_i.name = 'TOI'
        TOI_j.name = 'TOI'
        TOI.name = 'TOI'

        print('Indirect TOI computed. TOI shape:',TOI.shape)
        print('  Time elapsed:',datetime.now()-timer)
        return TOI, TOI_i, TOI_j


    def plot_system(self, unique_trips=False, transfer_stops=False, repr_stops_list=[], plot_stops=True, demand_col=False,
                    figsize=(20,15), annot=True):
        '''
        Description
        -----------

        Plot a geometric view of the the zones recorded in <shapes>.
        If <unique_trips> are given, plots the stops in <gtfs> and the line segments that connects them based on <unique_trips>.
        Also, the line width of the plotted unique trips corresponds to their (frequency*capacity) factor.
        If <transfer_stops> are given, plots the possible transfers between different unique_trips.
        If <repr_stops_list> are given, plots the representative stops of a specific zone.

        Parameters
        ----------

        * unique_trips {dict}
            If False, no unique trips are plotted.
            Otherwise, a dictionary containing all the unique trips in the keys and their corresponding (frequencies,capacities) in the values should be given

        * transfer_stops {False or dict}
            If False, no transfers are plotted.
            Otherwise, a dictionary containing the distinct pairs of unique trips and their corresponding transfer stops should be given

        * repr_stops {False or list}
            A list of zone_ids for which the representative stops will be highlighted

        * plot_stops {bool}
            If True, plot the stops on top of the zones of analysis

        * demand_col {column in shapes.columns}
            Numeric column in <shapes> that represents the demand for each zone of analysis and will be plotted using color coding

        * figsize {(int, int)}
            Figure size argument passed to matplotlib

        * annot {bool}
            If True, annotate the stops with their corresponding labels

        '''
        stops = self.filtered_gtfs['stops'].copy()
        stops_ = self.filtered_gtfs['stops'].set_index('stop_id')

        # Plotting the zones of analysis (optinally, color code using demand)
        if demand_col:
            self.shapes.plot(cmap='Blues',edgecolor='k', column=demand_col, legend=True, legend_kwds={'label':'demand'}, figsize=figsize)
        else:
            self.shapes.plot(cmap='Blues',edgecolor='k', figsize=figsize)

        # Plotting stops
        if plot_stops:
            for index, row in stops_.iterrows():
                plt.scatter(row['stop_lon'],row['stop_lat'])
                if annot:
                    plt.annotate(index, (row['stop_lon']+0.001,row['stop_lat']+0.001))

        # Plotting unique trips
        if unique_trips:
            if 'capacity' in self.filtered_gtfs.keys():
                avg = np.mean([freq*cap for (freq,cap) in unique_trips.values()])
                for sequence, (freq, cap) in unique_trips.items():
                    filtered_stops = stops_.loc[sequence,:]
                    plt.plot(filtered_stops.stop_lon, filtered_stops.stop_lat, linewidth=freq*cap*4/avg, alpha=0.7)
            else:
                avg = np.mean([freq for freq in unique_trips.values()])
                for sequence, freq in unique_trips.items():
                    filtered_stops = stops_.loc[sequence,:]
                    plt.plot(filtered_stops.stop_lon, filtered_stops.stop_lat, linewidth=freq*4/avg, alpha=0.7)

        # Plotting representative stops
        if repr_stops_list:
            repr_stops = self.repr_stops
            for zone_id in repr_stops_list:
                stops_lats, stops_lons = [],[]
                for (zone_id2, trip), (stop, dist) in repr_stops.items():
                    if zone_id2 == zone_id and stop:
                        stop_lat, stop_lon = stops_.loc[stop,['stop_lat','stop_lon']]
                        stops_lats.append(stop_lat)
                        stops_lons.append(stop_lon)
                plt.scatter(stops_lons, stops_lats, marker='v', c='red', s=150, label=f'Repr Stops of {zone_id}')
                zone_centroid = self.shapes.loc[zone_id,'geometry'].centroid
                plt.scatter(zone_centroid.x, zone_centroid.y, marker='*', c='black', s=300, label=f'Centroid of {zone_id}')
            plt.legend()

        # Plotting transfer stops
        if transfer_stops:
            for (stop_pair, dist) in transfer_stops.values():
                if stop_pair:
                    filter_ = stops.stop_id.apply(lambda x: x in stop_pair)
                    filtered_stops = stops[filter_]
                    plt.plot(filtered_stops.stop_lon, filtered_stops.stop_lat, linestyle='dashed')

        plt.show()
        
    def plot_TOI(self, toi, plot_arrow=False, figsize=(20,15), annot=True):
        '''
        Plots the TOI results on top of the geospatial plot of the zones of analysis
        '''
        if type(toi) == pd.Series:
            shapes = pd.concat((self.shapes,toi), axis=1)

            shapes.plot(cmap='YlGn',edgecolor='k', column='TOI', legend=True, legend_kwds={'label':'TOI'}, figsize=figsize)
            for index, row in shapes.iterrows():
                lon = row['geometry'].centroid.x
                lat = row['geometry'].centroid.y
                if annot:
                    plt.annotate(index, (lon,lat), fontsize='large', fontfamily='monospace')

        elif type(toi) == pd.DataFrame:
            # Plotting heatmap
            plt.figure(figsize=figsize)
            heatmap(toi, cmap='YlGn', cbar=False, annot=annot, fmt='.2f', linewidths=3)
            plt.title('TOI IJ')
            plt.show()

            if plot_arrow:
                # Plotting arrow diagram
                colors = {zone:np.random.rand(3) for zone in shapes.index}
                shapes = pd.concat((shapes, pd.Series(colors, name='color', index=shapes.index)), axis=1)

                shapes.plot(edgecolor='k', figsize=figsize, alpha=0.7, color='white')
                for index, row in shapes.iterrows():
                    lon = row['geometry'].centroid.x
                    lat = row['geometry'].centroid.y
                    if annot:
                        plt.annotate(index, (lon-0.002,lat+0.002), fontsize='large', fontfamily='monospace', color=colors[index])

                for zone_i in toi.index:
                    for zone_j in toi.columns:
                        v = toi.loc[zone_i,zone_j]
                        if v > 0:
                            lon_i = shapes.loc[zone_i,'geometry'].centroid.x
                            lat_i = shapes.loc[zone_i,'geometry'].centroid.y
                            lon_j = shapes.loc[zone_j,'geometry'].centroid.x
                            lat_j = shapes.loc[zone_j,'geometry'].centroid.y
                            # This draws an arrow from (x, y) to (x+dx, y+dy)
                            plt.arrow(x=lon_i, y=lat_i, dx=(lon_j-lon_i), dy=(lat_j-lat_i), alpha=0.7, width=v/toi.sum().sum()/100, color=colors[zone_i])
            plt.show()
            
    def geodesic2(self, v1, v2):    
        '''
        Computes the geodesic distance in kilometers between two locations by using the "Spherical Earth projected to a plane" formula. 
        This is a good approximation for small distances (relative to the radius of Earth) and is far quicker than geopy.geodesic.

        After testing:
            + geodesic2 can do 100,000 operations in 1.08 seconds
            + geodesic can do 100,000 operations in 12.6 seconds (~x12 slower)

        Source: https://en.wikipedia.org/wiki/Geographical_distance

        * v1 and v2 must each be a pair of (lat, lon) coordinates.

        '''
        v1_rad, v2_rad = np.array(v1)*np.pi/180, np.array(v2)*np.pi/180
        mean_lat, mean_lon = (v1_rad+v2_rad)/2
        delta_lat, delta_lon = v1_rad-v2_rad

        return 6371.009*np.sqrt((delta_lat)**2+(np.cos(mean_lat)*delta_lon)**2)

print('Succefully created shell for modelling!')
    
#-------------
# End of file
#-------------