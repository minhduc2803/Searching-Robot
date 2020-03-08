import math
from graph import *

def main():
    emap, world, S, G = init_world("input1.txt")

    print('Dijkstra algorithm')
    print('Size map: ',len(world[0])-1,'x',len(world)-1)

    display_world(world,S,G)

    print('searching...')

    start = time.time()

    total_g,shortest_path,recursive_display = dijkstra(emap, S, G)

    end = time.time()

    print('time searching: ',end-start,'(s)')

    if total_g == 0:
        print('can not find any path from S to G')
    else:
        print('total cost:',total_g)

    print('total nodes expand (opened and closed, maybe repeated):', len(recursive_display))

    n_show = int(len(recursive_display)/math.log(len(recursive_display),1.3))
    display_process(world,S,G,shortest_path,recursive_display,n_show)
    plt.show()
    
if __name__ == '__main__':
    main()