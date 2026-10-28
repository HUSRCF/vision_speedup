import math
from .native_graph_weight_cache_probe import CUTLASS,XMMA,XMMA_NONINDEXED

def reduced_grid(node,co,h,w):
    grid=list(node['grid']);name=node['name']
    if name==CUTLASS:
        full_tiles=math.ceil(16*h*w/128);half_tiles=math.ceil(8*h*w/128)
        assert grid[0]%full_tiles==0
        swizzle=grid[0]//full_tiles
        assert swizzle in (1,2,4,8) and grid[1]==math.ceil(math.ceil(co/128)/swizzle)
        grid[0]=half_tiles*swizzle;axis=0
    else:
        assert name.startswith((XMMA,XMMA_NONINDEXED)),name
        # XMMA's first tile dimension covers flattened batch/spatial rows;
        # its second dimension covers output channels. Earlier fixtures used
        # square tiles only. The remaining-shape inventory also witnessed the
        # nonindexed 256x128x32 family, with grid [ceil(CO/128),ceil(NHW/256),Z].
        tiles=tuple(map(int,name.split('tilesize')[1].split('_')[0].split('x')))
        assert tiles in ((64,64,64),(128,128,16),(256,128,32)),tiles
        spatial_tile,channel_tile,_=tiles
        assert node['grid'][0]==math.ceil(co/channel_tile) and node['grid'][1]==math.ceil(16*h*w/spatial_tile)
        grid[1]=math.ceil(8*h*w/spatial_tile);axis=1
    assert all(0<n<=o for n,o in zip(grid,node['grid'])) and grid[2]==node['grid'][2]
    return grid,axis
