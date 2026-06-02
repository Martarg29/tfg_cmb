import numpy as np
import healpy as hp


class Cut(object):
    '''
    cut a Healpix map to 12 parts, or to 12*subblocks_nums parts
    '''
    def __init__(self, maps_in, subblocks_nums=1, nest=False):
        '''
        :param maps_in: the input healpix maps (one map with the shape of (12*nside**2,) or multiple maps with the shape of (N, 12*nside**2))
        :param nest: bool, if False map_in is assumed in RING scheme, otherwise map_in is NESTED
        :param subblocks_nums: int, the number after dividing the square nside*nside into small squares,
                               subblocks_nums=1^2, 2^2, 4^2, 8^2, 16^2, ..., default 1
        '''
        self.maps_in = maps_in
        self.nest = nest
        self.subblocks_nums = subblocks_nums
    
    def _multi_map(self):
        return len(self.maps_in.shape) == 2
    
    def nside(self):
        m = self.maps_in[0] if self._multi_map() else self.maps_in
        return int(np.sqrt(m.shape[0] / self.subblocks_nums / 12))
    
    def _expand_array(self, original_array):
        '''
        to be used in nestedArray2nestedMap, expand the given small array into a large array
        
        :param original_array: with the shape of (2**n, 2**n), where n=1, 2, 4, 6, 8, 10, ...
        '''
        add_value = original_array.shape[0]**2
        a0 = original_array
        a1, a2, a3 = a0 + add_value, a0 + add_value*2, a0 + add_value*3
        return np.r_[np.c_[a3, a1], np.c_[a2, a0]]

    def _ordinal_array(self):
        '''
        obtain an array containing the ordinal number, the shape is (nside, nside)
        '''
        circle_num = (int(np.log2(self.nside()**2)) - 2) // 2
        ord_arr = np.array([[3.,1.],[2.,0.]])
        for _ in range(circle_num):
            ord_arr = self._expand_array(ord_arr)
        return ord_arr, circle_num

    def nestedArray2nestedMap(self, map_cut):
        '''
        reorder the cut map into NESTED ordering to show the same style using 
        plt.imshow() as that using Healpix
        
        :param map_cut: the cut map, the shape of map_cut is (nside**2,)
        
        return the reorded data, the shape is (nside, nside)
        '''
        array_fill, circle_num = self._ordinal_array()
        res = np.zeros_like(array_fill)
        for i in range(2**(circle_num+1)):
            for j in range(2**(circle_num+1)):
                res[i][j] = map_cut[int(array_fill[i][j])]
        return res.T

    def nestedMap2nestedArray(self, map_block):
        '''
        Restore the cut map(1/12 of full sky map) into an array which is in NESTED ordering
        
        need transpose if the map is transposed in nestedArray2nestedMap function
        '''
        map_block = map_block.T
        map_cut = np.zeros(self.nside()**2) # Cambiado a zeros
        array_fill, _ = self._ordinal_array()
        for i in range(array_fill.shape[0]):
            for j in range(array_fill.shape[1]):
                map_cut[int(array_fill[i][j])] = map_block[i][j]
        return map_cut
    
    def _block(self, Map, block_n):
        map_NEST = Map if self.nest else hp.reorder(Map, r2n=True)
        start = block_n * self.nside()**2
        end = (block_n + 1) * self.nside()**2
        return self.nestedArray2nestedMap(map_NEST[start:end])
    
    def block_all(self):
        if self._multi_map():
            return [[self._block(m, b) for b in range(12)] for m in self.maps_in]
        return [self._block(self.maps_in, b) for b in range(12)]


class Block2Full(Cut):
    '''
    stitch a cut map (1/12 of full sky map) to a full sky map with other parts is zeros
    '''
    def __init__(self, maps_block, block_n, base_map=None, nest=False):
        '''
        :param maps_block: the cut map in NESTED ording (one map in 2D array with the shape of (nside, nside) 
        or multiple maps in 3D array with the shape of (N, nside, nside) or multiple maps in a list with each element has the shape of (nside,nside))
        :param block_n: int, the number of cut map, 0, 1, 2, ..., 11
        :param nest: bool, if False base_map is assumed in RING scheme, otherwise base_map is NESTED
        :param subblocks_nums: int, the number after dividing the square nside*nside into small squares,
                               subblocks_nums=1^2, 2^2, 4^2, 8^2, 16^2, ..., default 1
        '''
        self.maps_block = np.array(maps_block)
        self.block_n = block_n
        self.base_map = base_map
        self.nest = nest
        self.subblocks_nums = 1
    
    def _full(self, map_block):
        '''
        return a full sphere map
        '''
        self.temp_nside = map_block.shape[0]
        arr = self.nestedMap2nestedArray(map_block)
        m_nest = np.zeros(12 * self.temp_nside**2) if self.base_map is None else self.base_map
        m_nest[self.block_n * self.temp_nside**2 : (self.block_n+1) * self.temp_nside**2] = arr
        return hp.reorder(m_nest, n2r=True)
    
    def full(self):
        if len(self.maps_block.shape) == 3: # Multiples bloques
            return np.array([self._full(b) for b in self.maps_block])
        return self._full(self.maps_block)


#%% piece plane map
def sphere2piecePlane(sphere_map):
    '''
    cut full map to 12 blocks, then piecing them together into a plane
    this is only for the case of subblocks_nums=1
    '''    
    ct = Cut(sphere_map)
    blks = ct.block_all()
    n = ct.nside()
    res = np.zeros((n*4, n*3))
    # Mapeo optimizado de los 12 parches de Healpix
    idx_map = [(1,3,0), (5,3,1), (8,3,2), (0,2,0), (4,2,1), (11,2,2), 
               (3,1,0), (7,1,1), (10,1,2), (2,0,0), (6,0,1), (9,0,2)]
    for b_idx, row, col in idx_map:
        res[row*n:(row+1)*n, col*n:(col+1)*n] = blks[b_idx]
    return res

def anafast_spectra(Map, Map2=None, lmax=None, gal_cut=0, is_fullMap=True, block_n=4,
                    ell_start=2):
    """
    ell_start: 0 or 2. If 0, the output \ell start from 0, if 2, the output \ell start from 2. Default: 2
    """
    Cl = hp.anafast(Map, lmax=lmax)
    ell = np.arange(len(Cl))
    Dl = Cl * ell * (ell + 1) / (2 * np.pi)
    return ell[2:], Dl[2:]


#%% pixel size
class PixelSize:
    def __init__(self, nside):
        self.nside = nside
    def report(self):
        ps = (360**2 / np.pi) / (12 * self.nside**2)
        print(f"Resolución: {np.sqrt(ps)*60:.2f} arcmin por píxel")
