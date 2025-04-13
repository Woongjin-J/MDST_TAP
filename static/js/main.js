// Configure the API URL based on the environment
const API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? '' // Use relative URL for local development
    : 'http://overpass-api.de/api/interpreter'; // Replace with your actual backend URL

var map = L.map('map').setView([37.0902, -95.7129], 4);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

var currentMarker = null;
var features = {
    'Traffic_Signal_Flag': 0,
    'Crossing_Flag': 0,
    'Highway_Flag': 1,
    'Distance(mi)': 0.1,
    'Start_Hour': 12,
    'Start_Month': 6,
    'Accident_Duration': 5
};

let updateTimer = null;

function showError(message) {
    const errorDiv = document.getElementById('error-message');
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        setTimeout(() => {
            errorDiv.style.display = 'none';
        }, 5000);
    }
    console.error(message);
}

function showLoading(show = true) {
    const loadingElement = document.getElementById('loading');
    if (loadingElement) {
        if (show) {
            loadingElement.classList.add('show');
        } else {
            loadingElement.classList.remove('show');
        }
    }
}

function updateRoadTypeDisplay(roadType, roadName) {
    const highwayToggle = document.getElementById('Highway_Flag');
    if (highwayToggle) {
        const newValue = roadType === 'Highway' ? 1 : 0;
        highwayToggle.checked = newValue === 1;
        features['Highway_Flag'] = newValue;
    }
}

// Update toggle values and handle changes
document.querySelectorAll('.toggle-switch input').forEach(toggle => {
    toggle.addEventListener('change', function() {
        features[this.id] = this.checked ? 1 : 0;
        if (updateTimer) {
            clearTimeout(updateTimer);
        }
        if (currentMarker) {
            updateTimer = setTimeout(() => {
                updatePrediction(currentMarker.getLatLng());
            }, 500);
        }
    });
});

// Update slider values display and handle changes
document.querySelectorAll('.slider').forEach(slider => {
    const valueDisplay = document.getElementById(slider.id + '_value');
    if (valueDisplay) {
        valueDisplay.textContent = slider.value;
    }

    slider.addEventListener('input', function() {
        const value = this.value;
        if (valueDisplay) {
            valueDisplay.textContent = value;
        }
        features[this.id] = parseFloat(value);
        if (updateTimer) {
            clearTimeout(updateTimer);
        }
        if (currentMarker) {
            updateTimer = setTimeout(() => {
                updatePrediction(currentMarker.getLatLng());
            }, 500);
        }
    });
});

// Handle map clicks
map.on('click', function(e) {
    if (currentMarker) {
        map.removeLayer(currentMarker);
    }
    currentMarker = L.marker(e.latlng);
    currentMarker.addTo(map);
    updatePrediction(e.latlng);
});

async function updatePrediction(latlng) {
    try {
        showLoading(true);

        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            mode: 'cors',
            body: JSON.stringify({
                ...features,
                lat: latlng.lat,
                lon: latlng.lng
            })
        });

        showLoading(false);

        if (!response.ok) {
            throw new Error(`Server returned ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        const severity = data.severity;
        const severityText = ['Low', 'Medium', 'High', 'Very High'][severity - 1];
        const severityClass = severity > 2 ? 'severity-high' : severity === 2 ? 'severity-medium' : 'severity-low';

        updateRoadTypeDisplay(data.road_type, data.road_name);

        const predictionElement = document.getElementById('prediction');
        if (predictionElement) {
            predictionElement.innerHTML = `
                <h3>Current Prediction</h3>
                <p>Location: ${latlng.lat.toFixed(4)}, ${latlng.lng.toFixed(4)}</p>
                <p>Road Type: <span class="${data.road_type === 'Highway' ? 'highway' : data.road_type === 'Local Road' ? 'local-road' : 'unknown-road'}">${data.road_type}</span></p>
                <p>Road Name: ${data.road_name}</p>
                <p>Severity: <span class="${severityClass}">${severityText}</span></p>
            `;
        }

        const popupContent = `
            <strong>Road Type:</strong> ${data.road_type}<br>
            <strong>Road Name:</strong> ${data.road_name}<br>
            <strong>Predicted Severity:</strong> ${severityText}
        `;
        currentMarker.bindPopup(popupContent).openPopup();
    } catch (error) {
        showLoading(false);
        console.error('Error:', error);
        showError(`Failed to get prediction: ${error.message}`);
        const predictionElement = document.getElementById('prediction');
        if (predictionElement) {
            predictionElement.innerHTML = `
                <h3>Error</h3>
                <p>Failed to get prediction. Please try again.</p>
            `;
        }
    }
}