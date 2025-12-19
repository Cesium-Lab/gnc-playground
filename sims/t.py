import numpy as np

class E:
    def __init__(self, v):
        self.v = v


arr = np.array([4])

a = E(arr)

print(a.v)

arr[0] += 1

print(a.v)
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
# are the lat/lon values of the lower left and upper right corners
# of the map.
# resolution = 'c' means use crude resolution coastlines.
m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='i')
m.drawcoastlines()
m.fillcontinents(color="#00BE00",lake_color='aqua')
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,30.))
m.drawmeridians(np.arange(-180.,181.,60.))
m.drawmapboundary(fill_color='aqua') 
plt.title("Equidistant Cylindrical Projection")
plt.show()
