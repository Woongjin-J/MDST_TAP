import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.widgets import Slider

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
model.load_state_dict(torch.load('best_accident_severity_model.pth'))
model.eval()

features = {
    'Traffic_Signal_Flag': 0,
    'Crossing_Flag': 0,
    'Highway_Flag': 1,
    'Distance(mi)': 1.0,  # Default distance (normalized)
    'Start_Hour': 12,  # Default hour for the slider
    'Start_Month': 6,  # Default month for the slider
    'Accident_Duration': 5  # Default duration in min (5 minutes)
}


def transform_features(raw):
    return {
        'Traffic_Signal_Flag': raw['Traffic_Signal_Flag'],
        'Crossing_Flag': raw['Crossing_Flag'],
        'Highway_Flag': raw['Highway_Flag'],
        'Distance(mi)': raw['Distance(mi)'],
        'Start_Hour_Sin': np.sin(2 * np.pi * raw['Start_Hour'] / 24),
        'Start_Hour_Cos': np.cos(2 * np.pi * raw['Start_Hour'] / 24),
        'Start_Month_Sin': np.sin(2 * np.pi * raw['Start_Month'] / 12),
        'Start_Month_Cos': np.cos(2 * np.pi * raw['Start_Month'] / 12),
        'Accident_Duration': (raw['Accident_Duration'] - 315.48991736325416) / 9888.00371222839
    }


def predict_severity():
    transformed = transform_features(features)
    x = torch.tensor([list(transformed.values())], dtype=torch.float32)
    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1).item() + 1
        return pred

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(11, 6))
ax.set_extent([-125, -65, 24, 50], crs=ccrs.PlateCarree())

# Style the map
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.5)
ax.add_feature(cfeature.BORDERS, edgecolor='black')
ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.5)
ax.add_feature(cfeature.COASTLINE)

# Initial title
ax.set_title("Click on a location to predict severity", fontsize=14, pad=20)

red_dot = None

def on_click(event):
    global red_dot
    if event.inaxes == ax:
        if red_dot:
            red_dot.remove()
        red_dot = ax.plot(event.xdata, event.ydata, 'ro', markersize=6, transform=ccrs.Geodetic())[0]
        severity = predict_severity()
        ax.set_title(f"Clicked: ({event.xdata:.2f}, {event.ydata:.2f}) | Predicted Severity: {severity}", fontsize=14, pad=20)
        fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', on_click)

slider_ax = []
sliders = []
slider_names = list(features.keys())

for i, name in enumerate(slider_names):
    ax_slider = plt.axes([0.15, 0.02 + i*0.03, 0.65, 0.02])
    if 'Flag' in name:
        slider = Slider(ax_slider, name, 0, 1, valinit=features[name], valstep=1)
    elif name == 'Start_Hour':
        slider = Slider(ax_slider, name, 0, 23, valinit=12, valstep=1)
    elif name == 'Start_Month':
        slider = Slider(ax_slider, name, 1, 12, valinit=6, valstep=1)
    elif name == 'Distance(mi)':
        slider = Slider(ax_slider, name, 0, 10.0, valinit=1.0)
    elif name == 'Accident_Duration':
        slider = Slider(ax_slider, name, 1, 180, valinit=5)  # in minutes: 1 min to 3 hrs
    else:
        slider = Slider(ax_slider, name, -1, 1, valinit=features[name])
    slider_ax.append(ax_slider)
    sliders.append(slider)

def update(val):
    for i, name in enumerate(slider_names):
        features[name] = sliders[i].val

    # features = transform_features(features)

    if red_dot:
        severity = predict_severity()
        ax.set_title(f"Predicted Severity: {severity}", fontsize=14, pad=20)
        fig.canvas.draw_idle()

for s in sliders:
    s.on_changed(update)

plt.subplots_adjust(top=0.88, bottom=0.35)  # Adjust to give room for the title
plt.show()
