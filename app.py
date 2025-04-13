import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
import os
import socket
import webbrowser
from threading import Timer
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_port_available(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1', port))
        available = True
    except:
        available = False
    sock.close()
    return available

def find_available_port(start_port=8080):
    port = start_port
    while not check_port_available(port) and port < start_port + 20:
        port += 1
    return port if port < start_port + 20 else None

# Define the neural network model for accident severity prediction
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

# Load the pre-trained model weights
try:
    model = AccidentSeverityModel()
    model.load_state_dict(torch.load('best_accident_severity_model.pth'))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure 'accident_severity_model.pth' is in the current directory.")
    exit(1)

# Create Flask app
app = Flask(__name__, static_url_path='')
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins for now
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Origin"],
        "expose_headers": ["Access-Control-Allow-Origin"],
        "supports_credentials": True
    }
})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def get_road_info(lat, lon):
    try:
        overpass_url = "http://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        way(around:50,{lat},{lon})
        ["highway"];
        out body;
        >;
        out body qt;
        """

        logger.info(f"Querying Overpass API for location: {lat}, {lon}")
        response = requests.post(overpass_url, data=query)

        if response.status_code != 200:
            logger.warning(f"Overpass API returned status code: {response.status_code}")
            return {'type': 'Unknown', 'name': 'Unknown'}

        data = response.json()

        if not data.get('elements'):
            logger.info("No roads found nearby")
            return {'type': 'Unknown', 'name': 'Unknown'}

        roads = []
        for element in data['elements']:
            if 'tags' in element and 'highway' in element['tags']:
                road_info = {
                    'highway_type': element['tags']['highway'],
                    'name': element['tags'].get('name', 'Unnamed Road'),
                    'ref': element['tags'].get('ref', '')
                }
                roads.append(road_info)

        logger.info(f"Found roads: {roads}")

        highway_categories = {
            'motorway': 'Highway',
            'trunk': 'Highway',
            'motorway_link': 'Highway',
            'trunk_link': 'Highway',
            'primary': 'Highway',
            'primary_link': 'Highway',
            'secondary': 'Highway',
            'secondary_link': 'Highway'
        }

        for road in roads:
            if road['highway_type'] in highway_categories:
                road_name = road['ref'] if road['ref'] else road['name']
                return {'type': 'Highway', 'name': road_name}

        if roads:
            road = roads[0]
            return {'type': 'Local Road', 'name': road['name']}

        return {'type': 'Unknown', 'name': 'Unknown'}

    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying Overpass API: {str(e)}")
        return {'type': 'Unknown', 'name': 'Unknown'}
    except Exception as e:
        logger.error(f"Unexpected error in get_road_info: {str(e)}")
        return {'type': 'Unknown', 'name': 'Unknown'}

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

@app.route('/')
def home():
    try:
        return send_file('index.html')
    except Exception as e:
        logger.error(f"Error loading index.html: {str(e)}")
        return f"Error: Could not load the HTML file. {str(e)}", 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.json
        lat, lon = data.get('lat'), data.get('lon')

        road_info = {'type': 'Unknown', 'name': 'Unknown'}
        if lat is not None and lon is not None:
            logger.info(f"Getting road info for coordinates: {lat}, {lon}")
            road_info = get_road_info(lat, lon)
            logger.info(f"Detected road info: {road_info}")
            data['Highway_Flag'] = 1 if road_info['type'] == 'Highway' else 0

        transformed = transform_features(data)
        x = torch.tensor([list(transformed.values())], dtype=torch.float32)
        with torch.no_grad():
            output = model(x)
            pred = torch.argmax(output, dim=1).item() + 1

        response_data = {
            'severity': pred,
            'road_type': road_info['type'],
            'road_name': road_info['name'],
            'is_highway': road_info['type'] == 'Highway'
        }
        logger.info(f"Prediction response: {response_data}")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    port = find_available_port()
    if port:
        print(f"Server will be available at http://127.0.0.1:{port}")
        print("Opening browser automatically...")
        Timer(1.5, lambda: webbrowser.open(f'http://127.0.0.1:{port}/')).start()
        try:
            app.run(host='127.0.0.1', port=port, debug=False)
        except Exception as e:
            print(f"Error starting server: {e}")
    else:
        print("No available port found. Please try again later.")