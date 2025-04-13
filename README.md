# Accident Severity Prediction Map

A web application that predicts accident severity based on location and various factors using machine learning.

## Features

- Interactive map interface
- Real-time accident severity prediction
- Road type detection using OpenStreetMap API
- Adjustable parameters:
  - Traffic signal presence
  - Crossing presence
  - Highway status
  - Distance
  - Start hour
  - Start month
  - Accident duration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/accident-severity-prediction.git
cd accident-severity-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install flask flask-cors torch numpy folium requests
```

4. Download the pre-trained model:
- Place the model file (`best_accident_severity_model.pth`) in the project root directory

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://127.0.0.1:8080
```

3. Click on the map to get predictions for specific locations.

## Project Structure

```
├── app.py              # Main application file
├── index.html          # Main page template
├── static/            # Static files
│   ├── css/          # CSS styles
│   └── js/           # JavaScript files
└── README.md          # Project documentation
```

## Dependencies

- Flask
- Flask-CORS
- PyTorch
- NumPy
- Folium
- Requests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

