
from pykml import parser
import re

import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch

import numpy as np
import scipy.io

kml_file = 'community_areas.kml'
result_file = 'community_areas.mat'

def get_boundaries():
    """ Parse the KML file to get the boundaries
    """
    with open(kml_file) as f:
        tree = parser.parse(f)
        root = tree.getroot()

        N = 0
        placemarks = {}
        for ch in root.Document.Folder.getchildren():
            if 'Placemark' in ch.tag:
                # Found a placemark
                N += 1
                pname = int(ch.name.text)
                
                # Get the coordinates
                pcoords = ch.MultiGeometry.Polygon.outerBoundaryIs.LinearRing.coordinates.text
                pcoords = pcoords.strip()
                pcoords = re.split(',| ', pcoords)
                pcoords = [float(c) for c in pcoords]

                lons = pcoords[0::3]
                lats = pcoords[1::3]
                assert len(lons) == len(lats)
                #print "Polygon has %d vertices" % len(lons)
                
                placemarks[pname] = (lats, lons)

        print "Found %d placemarks" % N
        return placemarks

def convert_placemark_to_polygon(placemarks):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    polys = {}
    for (name, (lats,lons)) in placemarks.items():
        ext = zip(lons,lats)
        poly = Polygon(ext)
        
        # DEBUG - Plot the patch
        x,y = poly.exterior.xy
        ax.plot(x,y)
        
        ax.add_patch(PolygonPatch(poly, alpha=0.5))
    
        polys[name] = poly
        
    # plt.show()

    return polys

def get_area_of_polys(polys):
    """ Get the area of each of the polygons
    """
    K = len(polys)
    assert max(polys.keys()) == K, "ERROR: Expected one key per poly"
    assert min(polys.keys()) == 1, "ERROR: Expected one key per poly"

    areas = np.zeros(len(polys))
    for k in np.arange(K):
        areas[k] = polys[k+1].area
        print "Region %d:\tArea:%f" % (k,areas[k])
    return areas

def save_results(polys, areas):
    """ Save the areas to a matlab array
    """
    import pdb; pdb.set_trace()
    K = len(polys)
    bounds = []
    for k in np.arange(K):
        x,y = polys[k+1].exterior.xy
        bounds.append(np.vstack((x,y)).T)
    res = {'areas' : areas, 'polys' : bounds}
    scipy.io.savemat(result_file, res)

def parse_community_areas():
    placemarks = get_boundaries()
    polys = convert_placemark_to_polygon(placemarks)
    areas = get_area_of_polys(polys)
    save_results(polys, areas)

if __name__ == '__main__':
    parse_community_areas()
