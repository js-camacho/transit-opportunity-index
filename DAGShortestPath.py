# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:59:10 2020

@author: johan
"""
#===============================================
# Shortest Path in Directed Acyclic Graphs (DAG)
#===============================================
''' Documentation
--------
Sources:
--------
About algorithm:        https://www.geeksforgeeks.org/shortest-path-for-directed-acyclic-graphs/
Topological Sorting:    https://www.geeksforgeeks.org/topological-sorting/

-------------
Description:
-------------
     We can calculate single source shortest distances in O(|V|+|E|) time for DAGs, which is faster than Dijkstra. 
     The idea is to use Topological Sorting into a Dijkstra algorithm. Topological Sorting is a linear ordering 
     of vertices such that for every directed edge (u,v), vertex u comes before v in the ordering.
     We one by one process all vertices in topological order. 
     For every vertex being processed, we update distances of its adjacent using distance of current vertex.
    
---------------
The algorithm:
---------------
    1. Initialize all distances to source as infinity excpet for source node, which is 0.
    2. Create a Topological Order of all vertices
    3. For every vertex <u> in the topological order:
        3.1. For every vertex <v> adjacent to <u>:
            3.1.1. If distance(<v>) > distance(<u>) + weight(<u>,<v>): 
                    distance(<v>) = distance(<u>) + weight(<u>,<v>)
 
------------
Complexity:
------------
(|V|: number of vertices, |E|: number of edges)
    * Bellman-Ford (for general weighted graph): O(|V|*|E|)
    * Dijkstra (for non-negative weighted graph): O(|V|*log|V| + |E|)
    * DAG with Topological Sorting (only for DAG): O(|V|+|E|)
'''
'''
Create a Graph using adjacency list:
Every node of adjacency list contains vertex number of the vertex to which edge connects. 
It also contains weight of the edge.
-----------
Parameters:
-----------
* vertices : int
   Number of vertices in the graph
'''
#%% Importing packages

from collections import defaultdict 
from datetime import datetime

#%% 
# Python program to find single source shortest paths 
# for Directed Acyclic Graphs Complexity :OV(V+E) 

  
# Graph is represented using adjacency list. Every 
# node of adjacency list contains vertex number of 
# the vertex to which edge connects. It also contains 
# weight of the edge

class DAG: 
    def __init__(self, n_vertices): 
        '''
        Description
        -----------
        
        Create a Directed Acyclic graph shell with the specified number of vertices
        
        Parameters
        ----------
        
        - n_vertices: int
            Number of vertices in the graph. The vertices will be zero-indexed.
        
        '''
        self.n_vertices = n_vertices # Number of vertices  
        self.adj = defaultdict(list) # Dictionary containing adjacency List
  

    def add_edge(self, u, v, weight): 
        '''
        Description
        -----------
        
        Add an edge to the DAG
        
        Parameters
        ----------
        
        - u: int in range(n_vertices)
            Tail of the edge
        - v: int in range(n_vertices)
            Head of the edge
        - weight: float >= 0
            Edge weight to be minimized
            
        '''
        self.adj[u].append((v,weight)) 
  

    def topological_sort_util(self, u, visited, stack): 
        '''
        Description
        -----------
        
        Recursive function to find the Topological Sort of the vertices.
        Used by shortest_path().
        
        Parameters
        ----------
        
        - u: int in range(n_vertices)
            Vertex to be explored
        - visited: bool
            Wheter the vertex has already been called by this function before or not
        - stack: list
            Current topological sort in reverse order
        '''
  
        # Mark the current node as visited. 
        visited[u] = True
        
        # Recur for all the vertices adjacent to this vertex 
        if u in self.adj.keys(): 
            for v, weight in self.adj[u]: 
                if visited[v] == False: 
                    self.topological_sort_util(v, visited, stack) 
  
        # Push current vertex to stack which stores topological sort 
        stack.append(u)
  

    def shortest_path(self, source, target=None): 
        '''
        Description
        -----------
        
        Find shortest paths from a given (source) vertex to all the other vertices in the DAG.
        It uses topological_sort_util() method.
        
        Parameters
        ----------
        
        - source: int in range(n_vertices)
            Vertex from which the shortest paths will be generated
        
        - target: int in range(n_vertices) or None
            Optionally, the algorithm can stop early after finding the shortest path for a specific vertex
        '''
        # Mark all the vertices as not visited 
        visited = [False]*self.n_vertices 
        stack =[]
        predecessors = {i: None for i in range(self.n_vertices)}
  
        # Call the recursive helper function to store Topological Sort starting from source vertice 
        for i in range(self.n_vertices): 
            if visited[i] == False: 
                self.topological_sort_util(source, visited, stack) 
  
        # Initialize distances to all vertices as infinite and distance to source as 0 
        dist = [float("Inf")] * (self.n_vertices) 
        dist[source] = 0
  
        # Process vertices in topological order 
        while stack: 
  
            # Get the next vertex from topological order 
            u = stack.pop() 
  
            # Update distances of all adjacent vertices 
            for v, weight in self.adj[u]: 
                if dist[v] > dist[u] + weight: 
                    predecessors[v] = u
                    dist[v] = dist[u] + weight
            
            if u == target:
                break
  
        # Get the resulting paths and distances
        paths = {}
        for u in range(self.n_vertices):
            path = []
            pred = u
            while pred != None:
                pred = predecessors[pred]
                if pred == None:
                    break
                path.insert(0, pred)
            paths[u] = [path+[u], dist[u]]
        
        return paths