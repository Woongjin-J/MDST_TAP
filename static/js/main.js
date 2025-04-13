var map = L.map('map').setView([37.0902, -95.7129], 4);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

var currentMarker = null;
var features = {
    'Traffic_Signal_Flag': 0,
    'Crossing_Flag': 0,
    'Highway_Flag': 1,
    'Distance(mi)': 1.0,
    'Start_Hour': 12,
    'Start_Month': 6,
    'Accident_Duration': 5
};

let updateTimer = null;

function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}

function updateRoadTypeDisplay(roadType, roadName) {
    // Update Highway Flag toggle only
    const highwayToggle = document.getElementById('Highway_Flag');
    const newValue = roadType === 'Highway' ? 1 : 0;
    highwayToggle.checked = newValue === 1;
    features['Highway_Flag'] = newValue;
}

// Update toggle values and handle changes
document.querySelectorAll('.toggle-switch input').forEach(toggle => {
    toggle.addEventListener('change', function() {
        features[this.id] = this.checked ? 1 : 0;

        // Clear any existing timer
        if (updateTimer) {
            clearTimeout(updateTimer);
        }

        // Set a new timer to update prediction after 500ms
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
    valueDisplay.textContent = slider.value;

    slider.addEventListener('input', function() {
        const value = this.value;
        valueDisplay.textContent = value;
        features[this.id] = parseFloat(value);

        // Clear any existing timer
        if (updateTimer) {
            clearTimeout(updateTimer);
        }

        // Set a new timer to update prediction after 500ms of no slider movement
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
        // Show loading indicator
        const loadingElement = document.getElementById('loading');
        loadingElement.classList.add('show');

        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                ...features,
                lat: latlng.lat,
                lon: latlng.lng
            })
        });

        // Hide loading indicator
        loadingElement.classList.remove('show');

        if (!response.ok) {
            throw new Error('Server returned an error');
        }

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        const severity = data.severity;
        const severityText = ['Low', 'Medium', 'High', 'Very High'][severity - 1];
        const severityClass = severity > 2 ? 'severity-high' : severity === 2 ? 'severity-medium' : 'severity-low';

        // Update Highway Flag based on road type
        updateRoadTypeDisplay(data.road_type, data.road_name);

        // Update prediction display
        document.getElementById('prediction').innerHTML = `
            <h3>Current Prediction</h3>
            <p>Location: ${latlng.lat.toFixed(4)}, ${latlng.lng.toFixed(4)}</p>
            <p>Road Type: <span class="${data.road_type === 'Highway' ? 'highway' : data.road_type === 'Local Road' ? 'local-road' : 'unknown-road'}">${data.road_type}</span></p>
            <p>Road Name: ${data.road_name}</p>
            <p>Severity: <span class="${severityClass}">${severityText}</span></p>
        `;

        const popupContent = `
            <strong>Road Type:</strong> ${data.road_type}<br>
            <strong>Road Name:</strong> ${data.road_name}<br>
            <strong>Predicted Severity:</strong> ${severityText}
        `;
        currentMarker.bindPopup(popupContent).openPopup();
    } catch (error) {
        // Hide loading indicator on error
        document.getElementById('loading').classList.remove('show');
        console.error('Error:', error);
        showError(`Failed to get prediction: ${error.message}`);
        document.getElementById('prediction').innerHTML = `
            <h3>Error</h3>
            <p>Failed to get prediction. Please try again.</p>
        `;
    }
}