import math
from graph import *

def main():
    world, emap, middle_node, S, G = init_world('input31.txt',True)

    print('Level 3 searching')
    print('Size map: ',len(world[0])-1,'x',len(world)-1)

    display_world(world,S,G,lv3=True,middle_node=middle_node)
    #plt.show()
    print('searching...')

    start = time.time()
    
    total_g, shortest_path, recursive_display, shortest_paths, p = lv3_search(emap, S, G, middle_node)

    end = time.time()

    print('time searching: ',end-start,'(s)')

    if total_g == 0:
        print('can not find any path from S to G via all middle nodes')
    else:
        print('total cost:',total_g)
        via_result = 'via S -> A'
        for i in p[:-1]:
            via_result = via_result+str(i)+'('+str(middle_node[i][0])+';'+str(middle_node[i][1])+') -> A'
        via_result = via_result+str(p[-1])+'('+str(middle_node[p[-1]][0])+';'+str(middle_node[p[-1]][1])+') -> G'
        print(via_result)
    total_node = 0
    for display in recursive_display:
        for i in display:
            total_node = total_node + len(i)
    print('total nodes expand (opened and closed, maybe repeated):', total_node)

    n_show = int(total_node/math.log(total_node,1.5))
    display_process_lv3(world,S,G,middle_node,shortest_paths,shortest_path,recursive_display,n_show)
    

if __name__ == '__main__':
    main()