# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.feature import NaturalEarthFeature
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# class AccidentSeverityModel(nn.Module):
#     def __init__(self):
#         super(AccidentSeverityModel, self).__init__()
#         self.fc1 = nn.Linear(9, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 4)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)

#     def forward(self, x):
#         # TODO: Implement the forward pass using fc1, dropout
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# model = AccidentSeverityModel()
# model.load_state_dict(torch.load('accident_severity_model.pth'))
# model.eval()

# # Use default features for every prediction
# features = {
#     'Traffic_Signal_Flag': 0,
#     'Crossing_Flag': 0,
#     'Highway_Flag': 1,
#     'Distance(mi)': -0.3,
#     'Start_Hour_Sin': 0.0,
#     'Start_Hour_Cos': 1.0,
#     'Start_Month_Sin': 0.0,
#     'Start_Month_Cos': 1.0,
#     'Accident_Duration': -0.02
# }

# def predict_severity():
#     # TODO: Create a tensor from the feature values, pass it through the model, and return the predicted severity (add 1 to the output class index)
#     feature_tensor = torch.tensor(list(features.values()), dtype=torch.float32).unsqueeze(0)
#     with torch.no_grad():
#         output = model(feature_tensor)
#         _, predicted = torch.max(output, 1)
#         return predicted.item() + 1

# proj = ccrs.PlateCarree()
# fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(11, 6))
# ax.set_extent([-125, -65, 24, 50], crs=ccrs.PlateCarree())

# # TODO: Copy and paste what you did for week8_map_m.py
# # Add map features (land, ocean, lakes, borders, states, coastline)
# ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
# ax.add_feature(cfeature.LAND, facecolor='lightgray')
# ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
# ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)
# ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.5)
# ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.5)
# states = NaturalEarthFeature(category="cultural", name="admin_1_states_provinces_lines", scale="50m", facecolor="none")
# ax.add_feature(states, edgecolor='gray', linewidth=0.5, linestyle=':')

# ax.set_title("Click on a location to predict severity", fontsize=14, pad=20)

# red_dot = None

# def on_click(event):
#     # TODO: When user clicks on the map, show a red dot and update the title with predicted severity at that location
#     global red_dot
#     if event.xdata is not None and event.ydata is not None:
#         if red_dot:
#             red_dot[0].remove()
#         red_dot = ax.plot(event.xdata, event.ydata, 'ro', markersize=5, transform=ccrs.PlateCarree())
#         plt.draw()

#         severity = predict_severity()
#         ax.set_title(f"Clicked: ({event.xdata:.2f}, {event.ydata:.2f}) | Predicted Severity: {severity}", fontsize=14, pad=20)
#         plt.draw()

# fig.canvas.mpl_connect('button_press_event', on_click)

# plt.subplots_adjust(top=0.88)
# plt.show()








import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature

class AccidentSeverityModel(nn.Module):
    def __init__(self):
        super(AccidentSeverityModel, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = AccidentSeverityModel()
model.load_state_dict(torch.load('accident_severity_model.pth'))
model.eval()

features = {
    'Traffic_Signal_Flag': 0,
    'Crossing_Flag': 0,
    'Highway_Flag': 1,
    'Distance(mi)': 0.5,
    'Start_Hour_Sin': 0.0,
    'Start_Hour_Cos': 1.0,
    'Start_Month_Sin': 0.0,
    'Start_Month_Cos': 1.0,
    'Accident_Duration': 0.0
}

def predict_severity():
    x = torch.tensor([list(features.values())], dtype=torch.float32)
    with torch.no_grad():
        output = model(x)
        predicted = torch.argmax(output, dim=1).item() + 1
        return predicted

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(11, 6))
ax.set_extent([-125, -65, 24, 50], crs=ccrs.PlateCarree())

# Adding offline map features
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)
ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.5)
ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.5)

states = NaturalEarthFeature(category="cultural", name="admin_1_states_provinces_lines", scale="50m", facecolor="none")
ax.add_feature(states, edgecolor='gray', linewidth=0.5, linestyle=':')

ax.set_title("Click on a location to predict severity", fontsize=14, pad=20)
red_dot = None

def on_click(event):
    global red_dot
    if event.inaxes == ax:
        if red_dot:
            red_dot[0].remove()
        red_dot = ax.plot(event.xdata, event.ydata, 'ro', markersize=5, transform=ccrs.PlateCarree())

        severity = predict_severity()
        ax.set_title(f"Clicked: ({event.xdata:.2f}, {event.ydata:.2f}) | Predicted Severity: {severity}", fontsize=14, pad=20)
        plt.draw()

fig.canvas.mpl_connect('button_press_event', on_click)

# Adding Zoom Functions
def zoom(factor):
    x_min, x_max, y_min, y_max = ax.get_extent(crs=ccrs.PlateCarree())
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_range = (x_max - x_min) * factor / 2
    y_range = (y_max - y_min) * factor / 2

    new_extent = [x_center - x_range, x_center + x_range, y_center - y_range, y_center + y_range]
    ax.set_extent(new_extent, crs=ccrs.PlateCarree())
    plt.draw()

def zoom_in(event):
    zoom(0.8)

def zoom_out(event):
    zoom(1.2)

def on_scroll(event):
    if event.button == 'up':
        zoom(0.8)
    elif event.button == 'down':
        zoom(1.2)

fig.canvas.mpl_connect('scroll_event', on_scroll)

# Panning with Keyboard Arrow Keys
def on_key(event):
    x_min, x_max, y_min, y_max = ax.get_extent(crs=ccrs.PlateCarree())
    shift_amount = 2  # Adjust this value for how fast you want to pan

    if event.key == 'left':
        ax.set_extent([x_min - shift_amount, x_max - shift_amount, y_min, y_max], crs=ccrs.PlateCarree())
    elif event.key == 'right':
        ax.set_extent([x_min + shift_amount, x_max + shift_amount, y_min, y_max], crs=ccrs.PlateCarree())
    elif event.key == 'up':
        ax.set_extent([x_min, x_max, y_min + shift_amount, y_max + shift_amount], crs=ccrs.PlateCarree())
    elif event.key == 'down':
        ax.set_extent([x_min, x_max, y_min - shift_amount, y_max - shift_amount], crs=ccrs.PlateCarree())

    plt.draw()

fig.canvas.mpl_connect('key_press_event', on_key)

plt.subplots_adjust(bottom=0.1)
plt.show()
