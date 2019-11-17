#!/usr/bin/env python
# coding: utf-8

# In[203]:
from time import sleep
from copy import deepcopy

import requests as re
import networkx as nx
from tqdm.autonotebook import tqdm, trange
from itertools import product, chain
from collections import defaultdict


# In[501]:

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np


# In[529]:


import requests
import json


# In[2]:


from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


color = SimpleNamespace(free=np.array([255, 255, 255], dtype=np.int),
                        wall=np.array([60, 60, 30], dtype=np.int),
                        car=np.array([30, 30, 200], dtype=np.int), #blue
                        orig=np.array([30, 200, 20], dtype=np.int),#green
                        dest=np.array([200, 20, 20], dtype=np.int)) #red
def visualize(width, height, gr, get_coord):
    img = np.ones(shape=(width, height, 3), dtype=np.int) * 255

    for i in range(width):
        for j in range(height):
            if not gr['grid'][i*width+j]:
                img[i, j, :] = color.wall    

    for user in gr['customers'].values():
        x, y = get_coord(user['origin'])
        img[x, y, :] = color.orig
        x, y = get_coord(user['destination'])
        img[x, y, :] = color.dest
        
    for car in gr['cars'].values():
        x, y = get_coord(car['position'])
        img[x, y, :] = color.car

    return img

url = "https://localhost:4321/rwh82g6djrgv8zo2vnmo8jlondntutf7/api/v1/actions"
headers = {
    'Authorization': "d4b3f4f5",
    'Content-Type': "application/x-www-form-urlencoded",
    'Accept': "*/*",
    'Cache-Control': "no-cache",
    'Host': "localhost:4321",
    'Accept-Encoding': "gzip, deflate",
    'Content-Length': "73",
    'Connection': "keep-alive",
    'cache-control': "no-cache"
}

def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    # Display dropped nodes.
    dropped_nodes = 'Dropped nodes:'
    d = 0
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if assignment.Value(routing.NextVar(node)) == node:
            dropped_nodes += ' {}'.format(manager.IndexToNode(node))
            d += 1
    print(dropped_nodes)
    print(f"Dropped {d} points")
    # Display routes
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print('Total Distance of all routes: {}m'.format(total_distance))
    print('Total Load of all routes: {}'.format(total_load))


INF = 10 ** 4 + 6

users_in_car = defaultdict(set)
users_storage = {}

def route(gr, plot=False):
    assert gr["width"] == gr["height"]

    w, h = gr["width"], gr["height"]

    grid = [gr["grid"][i:i + h] for i in range(0, len(gr["grid"]), w)]

    G = nx.grid_graph(dim=[w, h])



    for i, row in enumerate(grid):
        for j, e in enumerate(row):
            if not e:
                G.remove_node((i, j))
    #     G.add_node((i, j))




    cars, users = gr["cars"], gr["customers"]
    global users_storage
    if not users_storage:
        users_storage = deepcopy(users)

    capacities = [c['capacity'] for c in cars.values()] # prob: - c['used_capacity']

    for car_id, users_inside in users_in_car.items():
        for user in users_inside:
            users[user] = users_storage[user].copy()
            users[user]["origin"] = cars[car_id]["position"]

    def get_coord(el):
        return el // w, el % w


    coords = [get_coord(u['origin']) for u in users.values()] + [get_coord(u['destination']) for u in users.values()] + [get_coord(c['position']) for c in cars.values()]
    origin_dest_indices = [(i, i+len(users)) for i in range(len(users))]


    dist = [[0] * len(coords) for _ in range(len(coords))]
    for i in range(len(coords)):
        if coords[i] not in G:
            all_dists = {}
        else:
            all_dists = nx.algorithms.shortest_paths.shortest_path_length(G, coords[i])
        for j in range(len(coords)):
            dist[i][j] = all_dists.get(coords[j], INF)



    # %%time
    # dist = [[0] * len(coords) for _ in range(len(coords))]
    # for i in range(len(coords)):
    #     all_dists = nx.algorithms.shortest_paths.shortest_path(G, coords[i])
    #     for j in range(len(coords)):
    #         dist[i][j] = all_dists.get(coords[j], INF)

    cars_start = [i + len(users) * 2 for i in range(len(cars))]

    cars_end = [len(coords)] * len(cars)

    for i in range(len(coords)):
        dist[i].append(0)
    dist.append([INF] * (len(coords) + 1))


    demands = [1] * len(users) + [0] * (len(dist) - len(users))

    if plot:
        global fig_i
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(visualize(w, h, gr, get_coord))
        # plt.draw()
        # plt.pause(0.001)
        plt.savefig(f'img/{fig_i}.png')
        # plt.show(block=False)


    # # G OPT part


    def create_data_model():
        """Stores the data for the problem."""
        data = {}
        data['distance_matrix'] = dist
        data['demands'] = demands
        data['num_vehicles'] = len(cars)
        data['vehicle_capacities'] = capacities
        data['pickups_deliveries'] = origin_dest_indices
        data['starts'] = cars_start
        data['ends'] = cars_end
        return data


    data = create_data_model()


    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                            data['num_vehicles'],  data['starts'],
                                            data['ends'])
    routing = pywrapcp.RoutingModel(manager)


    # ### Distance

    def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]


    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)


    # ### Capacity

    # In[563]:


    def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]


    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')


    # ### Pickups and Deliveries


    for request in data['pickups_deliveries']:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(
                delivery_index))
        routing.solver().Add(
            distance_dimension.CumulVar(pickup_index) <=
            distance_dimension.CumulVar(delivery_index))


    # ### Penalties and Dropping Visits


    # Allow to drop nodes.
    penalty = 10000
    for node in range(len(users)):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
        
    for node in range(len(users), len(users) * 2):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)


    # ### Print


    # ### Solution


    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)


    search_parameters.time_limit.seconds = 5


    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print_solution(data, manager, routing, solution)
    else:
        print('Solution not found!')
        return


    for vehicle_id in range(data['num_vehicles']):
        if vehicle_id != fig_i % data['num_vehicles']:
            continue

        is_moved = False

        index = routing.Start(vehicle_id)
        while not is_moved:
            node_index = manager.IndexToNode(index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            to_node = manager.IndexToNode(index)
            print(node_index, to_node, vehicle_id)
            if routing.IsEnd(index):
                break
            is_moved = (coords[node_index] != coords[to_node])
        else:
            original_graph_path = nx.algorithms.shortest_paths.shortest_path(G, coords[node_index], coords[to_node])
            if len(original_graph_path) == 2:
                # add user to car
                capacity = 0
                for user_k, user_v in users.items():
                    vehicle_id_str = str(vehicle_id)
                    if get_coord(user_v["origin"]) == coords[to_node]:
                        if user_k not in users_in_car[vehicle_id_str]:
                            users_in_car[vehicle_id_str].add(user_k)
                            capacity += 1
                    if get_coord(user_v["destination"]) == coords[to_node]:
                        if user_k in users_in_car[vehicle_id_str]:
                            users_in_car[vehicle_id_str].discard(user_k)
                            capacity -= 1
                    if capacity == capacities[vehicle_id]:
                        break
                
            first_step = original_graph_path[1]
            move = None
            if first_step[0] > coords[node_index][0]:
                move = 0
            elif first_step[0] < coords[node_index][0]:
                move = 2
            elif first_step[1] > coords[node_index][1]:
                move = 1
            elif first_step[1] < coords[node_index][1]:
                move = 3
                
            action = {
                'carId': vehicle_id,
                'moveDirection': move
            } 
            
            body = {
                'type': 'move',
                'action': action
            }
            payload = json.dumps(body)


            response = requests.request("POST", url, data=payload, headers=headers, verify=False)
            print("sent response")
            # print(response.text)


def main():
    re.get('https://localhost:4321/rwh82g6djrgv8zo2vnmo8jlondntutf7/admin/start', verify=False)

    while True:  
        global fig_i
        fig_i += 1      
        response = re.get("https://localhost:4321/rwh82g6djrgv8zo2vnmo8jlondntutf7/api/v1/world",  verify=False)
        gr = response.json()
        route(gr, plot=True)
        sleep(0.5)


# if __name__ == "main":
print("Hey!")
# plt.ion()
fig_i = -1
main()
