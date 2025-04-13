# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from matplotlib.widgets import Slider

# proj = ccrs.PlateCarree()
# fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(11, 6))
# ax.set_extent([-125, -65, 24, 50], crs=ccrs.PlateCarree())

# # TODO: Add map features (land, ocean, lakes, borders, states, coastline)
# # ax.add_feature(cfeature.LAND, linewidth=0.3, facecolor='lightgray')
# # ax.add_feature(cfeature.OCEAN, linewidth=0.3, facecolor='lightblue')
# # ax.add_feature(cfeature.LAKES, linewidth=0.3, facecolor='lightblue')
# # ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
# # ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=':')
# # ax.add_feature(cfeature.COASTLINE)
# ax.set_facecolor(cfeature.COLORS['water'])
# ax.add_feature(cfeature.LAND)
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.BORDERS, linestyle='--')
# ax.add_feature(cfeature.LAKES, alpha=0.5)
# ax.add_feature(cfeature.STATES)
# ax.add_feature(cfeature.RIVERS)

# ax.set_title("Click on a location to predict severity", fontsize=14, pad=20)

# plt.subplots_adjust(top=0.88, bottom=0.35)
# plt.show()\



import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from matplotlib.widgets import Slider

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(11, 6))
ax.set_extent([-125, -65, 24, 50], crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)
ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.5)
ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.5)

# Add states using NaturalEarthFeature
states = NaturalEarthFeature(category="cultural", name="admin_1_states_provinces_lines", scale="50m", facecolor="none")
ax.add_feature(states, edgecolor='gray', linewidth=0.5, linestyle=':')

ax.set_title("Click on a location to predict severity", fontsize=14, pad=20)

plt.subplots_adjust(top=0.88, bottom=0.35)
plt.show()
