import matplotlib.pyplot as plt
import numpy as np
import time
import heapq
from itertools import permutations 

color_of_wall = [100,100,100]
color_of_polygon = [167, 171, 86]
color_of_frontier = [200,200,200]
color_of_closed_node = [175,175,175]
color_of_path = [255,255,0]
color_of_old_path = [186, 192, 254]
color_of_shortest_path = [249, 82, 58]
def find_edge(vertices, world, emap):
    vertices.append(vertices[0])
    vertices.append(vertices[1])
    vertices_row = [vertices[x] for x in range(1,len(vertices),2)]
    vertices_column = [vertices[x] for x in range(0,len(vertices),2)]
    for i in range(len(vertices_row)-1):
        dy = vertices_row[i+1]-vertices_row[i]
        dx = vertices_column[i+1]-vertices_column[i]
        sign_dx = 1
        if dx < 0:
            sign_dx = -1
        sign_dy = 1
        if dy < 0:
            sign_dy = -1
        if abs(dy) > abs(dx):
            dx = abs(dx/dy)*sign_dx
            dy = 1*sign_dy
        else:
            dy = abs(dy/dx)*sign_dy
            dx = 1*sign_dx
        y0 = vertices_row[i]
        x0 = vertices_column[i]
        xt,yt,xs,ys = x0,y0,x0,y0
        world[y0][x0] = color_of_polygon
        emap[y0][x0][0]= -1
        
        while xt != vertices_column[i+1] or yt != vertices_row[i+1]:
            x0 = x0 + dx
            y0 = y0 + dy
            xs = int(round(x0))
            ys = int(round(y0))
            world[ys][xs] = color_of_polygon
            emap[ys][xs][0] = -1
            if xs != xt and ys != yt:
                world[ys][xt] = color_of_polygon
                emap[ys][xt][0] = -1
            xt = xs
            yt = ys
    return world, emap

def init_world(file_name, lv3=False):
    f1 = open(file_name, "r")
    f = f1.readlines()
    temp = [int(x) for x in f[0].split(',')]
    width, height = temp[0]+1, temp[1]+1
    world = np.ones((height,width,3)).astype(int)*255
    emap = np.zeros((height,width,5)).astype(object)
    emap[:,:,0] = emap[:,:,0].astype(int)
    emap[:,:,2:] = emap[:,:,2:].astype(int)
    for x in range(width):
        world[0][x] = color_of_wall
        emap[0][x][0] = -1
    for x in range(width):
        world[height-1][x] = color_of_wall
        emap[height-1][x][0] = -1
    for x in range(height):
        world[x][0] = color_of_wall
        emap[x][0][0] = -1
    for x in range(height):
        world[x][width-1] = color_of_wall
        emap[x][width-1][0] = -1
    temp = [int(x) for x in f[1].split(',')]
    S = [temp[1],temp[0]]
    G = [temp[3], temp[2]]
    world[S[0]][S[1]] = np.array([245, 150, 0])
    world[G[0]][G[1]] = np.array([255, 0, 0])
    emap[G[0]][G[1]][-1] = 1
    temp = [int(x) for x in f[2].split(',')]
    n_polygon = temp[0]
    for ith in range(n_polygon):
        vertices = [int(x) for x in f[ith+3].split(',')]
        world, emap = find_edge(vertices, world, emap)
    if lv3 == False:
        return emap, world, S, G
    else:
        temp = [int(x) for x in f[n_polygon+3].split(',')]
        n_middle_node = temp[0]
        middle_node = []
        for i in range(n_middle_node):
            temp = [int(x) for x in f[i+4+n_polygon].split(',')]
            middle_node.append([temp[1],temp[0]])
            world[temp[1]][temp[0]] = color_of_wall
            emap[temp[1]][temp[0]][-1] = 1
        return world,emap,middle_node, S, G

def shortest_h(A, B):
    d_row = abs(A[0]-B[0])
    d_column = abs(A[1]-B[1])
    if d_row < d_column:
        return d_column + 0.5*d_row
    else:
        return d_row + 0.5*d_column

def zero(A,B):
    return 0

def display_world(world,S,G,lv3=False,middle_node=None):
    plt.clf()
    world[S[0]][S[1]] = np.array([245, 150, 0])
    world[G[0]][G[1]] = np.array([180, 0, 213])

    if lv3:
        for i in range(len(middle_node)):
            name = 'A'+str(i)
            plt.text(middle_node[i][1]-0.3,middle_node[i][0]-0.2,s=name,color='blue',weight='bold')

    plt.imshow(world,origin='lower',aspect='equal')
    plt.text(S[1]-0.3,S[0]-0.2,'S',color='blue',weight='bold')
    plt.text(G[1]-0.3,G[0]-0.2,'G',color='blue',weight='bold')
    
    plt.pause(0.001)
    
    
def display_process(world,S,G,shortest_path,recursive_display,n_show=100, lv3=False, middle_node=None):
    
    for i in range(1,len(recursive_display)-1):
        if recursive_display[i][1] == 2:
            world[recursive_display[i][0][0]][recursive_display[i][0][1]] = color_of_closed_node
        else:
            world[recursive_display[i][0][0]][recursive_display[i][0][1]] = color_of_frontier
        if i % n_show == 0 or i==len(recursive_display)-2:
            display_world(world,S,G,lv3,middle_node )
    for i in range(len(shortest_path)-1,-1,-1):
        world[shortest_path[i][0]][shortest_path[i][1]] = color_of_path
    display_world(world,S,G,lv3,middle_node)
    plt.show()

def check(l1,l2):
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            return False
    return True
def display_process_lv3(world, S, G, middle_node, shortest_paths, shortest_path, recursive_display, n_show):
    lv3 = True
    for display,paths in zip(recursive_display,shortest_paths):
        w = world.copy()
        for index in range(len(display)):
            
            for i in range(1,len(display[index])-1):
                if not check(w[display[index][i][0][0]][display[index][i][0][1]],color_of_old_path):
                    if display[index][i][1] == 2:
                        w[display[index][i][0][0]][display[index][i][0][1]] = color_of_closed_node
                    else:
                        w[display[index][i][0][0]][display[index][i][0][1]] = color_of_frontier
                    if i % n_show == 0 or i==len(display)-2:
                        display_world(w,S,G,lv3,middle_node)
            
            for i in range(len(paths[index])-1,-1,-1):
                w[paths[index][i][0]][paths[index][i][1]] = color_of_path
                world[paths[index][i][0]][paths[index][i][1]] = color_of_old_path
            display_world(w,S,G,lv3,middle_node)
    for item in shortest_path:
        for i in range(len(item)):
            world[item[i][0]][item[i][1]] = color_of_shortest_path
    display_world(world,S,G,lv3,middle_node)
    plt.show()


def heuristic_search(emap, S, G, heuristic, frontier=[]):
    recursive_display = []
    h_S = shortest_h(S,G)
    heapq.heappush(frontier,(h_S,S))
    d_child = [[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0]]
    while len(frontier) > 0:
        current_node = heapq.heappop(frontier)[1]
        emap[current_node[0]][current_node[1]][0] = -1
        if emap[current_node[0]][current_node[1]][-1] == 1:
            G = current_node
            break
        recursive_display.append([current_node,1])
        for d in d_child:
            child = [current_node[0]+d[0],current_node[1]+d[1]]
                
            if emap[child[0]][child[1]][0] != -1:
                c = 1
                if d[0] != 0 and d[1] != 0:
                    c = 1.5
                g = emap[current_node[0]][current_node[1]][1]+c
                if emap[child[0]][child[1]][0] == -2 and g >= emap[child[0]][child[1]][1]:
                    continue
                h = heuristic(child, G)
                f = g+h
                emap[child[0]][child[1]][0] = -2
                emap[child[0]][child[1]][1] = g
                emap[child[0]][child[1]][2], emap[child[0]][child[1]][3] = current_node[0],current_node[1]
                heapq.heappush(frontier,(f,child))
                recursive_display.append([child,2])

    shortest_path = []
    total_g = emap[G[0]][G[1]][1]
    if emap[G[0]][G[1]][2] != 0:
        p = [emap[G[0]][G[1]][2],emap[G[0]][G[1]][3]]
        while p[0] != S[0] or p[1] != S[1]:
            shortest_path.append(p)
            p = [emap[p[0]][p[1]][2],emap[p[0]][p[1]][3]]
    return total_g, shortest_path, recursive_display, frontier, G, emap

def A_star_search(emap, S, G):
    total_g, shortest_path, recursive_display, f, g, m = heuristic_search(emap, S, G, shortest_h)
    return total_g, shortest_path, recursive_display

def dijkstra(emap, S, G):
    total_g, shortest_path, recursive_display, f,g ,m = heuristic_search(emap, S, G, zero)
    return total_g, shortest_path, recursive_display

#emap[][] = [flag,h,parent_row,parent_column]
#flag = -1 => ko di dc
#flag = 0 => di duoc
#flag = 1 => goal
#flag = -2 => in frontier

def best_first_search(emap, S, G):
    recursive_display = []
    frontier = []
    h_S = shortest_h(S,G)
    heapq.heappush(frontier,(h_S,S))
    emap[S[0]][S[1]][0] = -1
    d_child = [[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0]]
    while len(frontier) > 0:
        current_node = heapq.heappop(frontier)[1]
        if current_node[0] == G[0] and current_node[1] == G[1]:
            break
        recursive_display.append([current_node,1])
        for d in d_child:
            child = [current_node[0]+d[0],current_node[1]+d[1]]
            
            if emap[child[0]][child[1]][0] != -1:
                emap[child[0]][child[1]][2],emap[child[0]][child[1]][3] = current_node[0],current_node[1]
                h = shortest_h(child,G)
                emap[child[0]][child[1]][0],emap[child[0]][child[1]][1] = -1,h
                heapq.heappush(frontier,(h,child))
                recursive_display.append([child,2])
                
    total_g = emap[G[0]][G[1]][1]
    shortest_path = []
    if emap[G[0]][G[1]][2] != 0:
        p = [emap[G[0]][G[1]][2],emap[G[0]][G[1]][3]]
        total_g = shortest_h(p,G)
        
        while p[0] != S[0] or p[1] != S[1]:
            shortest_path.append(p)
            parent = [emap[p[0]][p[1]][2],emap[p[0]][p[1]][3]]
            c = 1
            if p[0] != parent[0] and p[1] != parent[1]:
                c = 1.5
            total_g = total_g + c
            p = parent
    return total_g, shortest_path, recursive_display

class paths:
    def __init__(self, path, cost):
        self.path = path
        self.cost = cost
       

def find_node(node,middle_node):
    for i in range(len(middle_node)):
        if node[0] == middle_node[i][0] and node[1] == middle_node[i][1]:
            return i
    return -1

def find_all_path(emap, S, G, middle_node):
    #need shortest path from S to all middle node
    #save as [n,n] list; n = sizeof(middle_node)+1
    
    all_paths = []
    recursive_display = []
    shortest_paths = []
    emap[G[0]][G[1]] = 0
    e = emap.copy()
    frontier =  []
    
    
    all_path = [0 for x in range(len(middle_node))]
    display = []
    shortest_p = []
    for i in range(len(middle_node)):
        cost, path, display_temp, frontier, A, e = heuristic_search(e,S,G.copy(),zero,frontier)
        temp = paths(path,cost)
        
        display.append(display_temp)
        shortest_p.append(path)
        if cost==0:
            recursive_display.append(display)
            shortest_paths.append(shortest_p)
            return 0, all_paths, recursive_display,shortest_paths
        x = find_node(A,middle_node)
        
        all_path[x] = temp
    all_paths.append(all_path)
    recursive_display.append(display)
    shortest_paths.append(shortest_p)

    
    middle_node.append(G)
    emap[G[0]][G[1]] = 1
    
    for i in range(len(middle_node)-1):
        all_path = [0 for x in range(len(middle_node))]
        emap[middle_node[i][0]][middle_node[i][1]] = 0
        e = emap.copy()
        frontier = []
        display = []
        shortest_p = []
        for j in range(i+1,len(middle_node)):
            cost, path, display_temp, frontier, A, e = heuristic_search(e,middle_node[i],G.copy(),zero,frontier)
            temp = paths(path,cost)
            
            display.append(display_temp)
            shortest_p.append(path)
            if cost==0:
                recursive_display.append(display)
                shortest_paths.append(shortest_p)
                return 0, all_paths, recursive_display,shortest_paths
            x = find_node(A,middle_node)
            all_path[x] = temp
            
        all_paths.append(all_path)
        recursive_display.append(display)
        shortest_paths.append(shortest_p)
    middle_node.pop()
    return 1,all_paths, recursive_display,shortest_paths
    

def lv3_search(emap, S, G, middle_node):
    total_g_min,all_paths,recursive_display,shortest_paths = find_all_path(emap, S, G, middle_node)
    if total_g_min == 0:
        return total_g_min, [], recursive_display, shortest_paths, []
    total_g_min = 2*len(emap)*len(emap[0])
    L = list(permutations(range(0,len(middle_node))))
    min_p = L[0]
    
    for p in L:
        
        total_g = all_paths[0][p[0]].cost
        
        for i in range(1,len(p)):
            current = p[i-1]
            next = p[i]
            if p[i-1] > p[i]:
                current = p[i]
                next = p[i-1]
            total_g = total_g+all_paths[current+1][next].cost
            
        total_g = total_g+all_paths[p[-1]][-1].cost
        
        if total_g < total_g_min:
            total_g_min = total_g
            min_p = p
    shortest_path = [all_paths[0][min_p[0]].path]
    for i in range(1,len(min_p)):
        current = min_p[i-1]
        next = min_p[i]
        if min_p[i-1] > min_p[i]:
            current = min_p[i]
            next = min_p[i-1]
        shortest_path.append(all_paths[current+1][next].path)
    shortest_path.append(all_paths[min_p[-1]+1][-1].path)
    print(min_p)
    return total_g_min, shortest_path, recursive_display, shortest_paths, min_p


