from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import pickle
import random
import time
import threading
import json
from datetime import datetime
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Cho phép CORS để frontend có thể gọi API

class CNNLSTMModel(nn.Module):
    def __init__(self, input_features=10, sequence_length=10, num_classes=2):
        super(CNNLSTMModel, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=50, num_layers=2, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class DDoSDetector:
    def __init__(self, model_path='model.pth', sequence_length=10):
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_buffer = []
        
        # Load model hoặc tạo dummy model nếu không có file
        try:
            self.load_model()
        except:
            logger.warning("Model file not found. Using dummy model for demo.")
            self.model = CNNLSTMModel()
            self.model.eval()
        
        self.attack_types = {
            0: "Normal Traffic",
            1: "DDoS Attack"
        }
        
    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            input_features = checkpoint.get('input_features', 10)
            num_classes = checkpoint.get('num_classes', 2)
            self.model = CNNLSTMModel(input_features=input_features, 
                                    sequence_length=self.sequence_length,
                                    num_classes=num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model = CNNLSTMModel()
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
    def preprocess_features(self, raw_data):
        features = [
            raw_data.get('packet_length', 0),
            raw_data.get('protocol', 0),
            raw_data.get('source_port', 0),
            raw_data.get('dest_port', 0),
            raw_data.get('packet_rate', 0),
            raw_data.get('byte_rate', 0),
            raw_data.get('flow_duration', 0),
            raw_data.get('packet_count', 0),
            raw_data.get('unique_sources', 0),
            raw_data.get('unique_destinations', 0)
        ]
        
        # Simple normalization
        features = np.array(features, dtype=np.float32)
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        return features
    
    def create_sequence(self, new_features):
        self.data_buffer.append(new_features)
        if len(self.data_buffer) > self.sequence_length:
            self.data_buffer.pop(0)
        
        if len(self.data_buffer) < self.sequence_length:
            padding_needed = self.sequence_length - len(self.data_buffer)
            padded_data = [np.zeros_like(new_features)] * padding_needed + self.data_buffer
        else:
            padded_data = self.data_buffer
        
        return np.array(padded_data)
    
    def predict(self, raw_data):
        try:
            features = self.preprocess_features(raw_data)
            sequence_data = self.create_sequence(features)
            input_tensor = torch.FloatTensor(sequence_data).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            prediction_label = self.attack_types.get(predicted.item(), "Unknown")
            confidence_score = confidence.item()
            
            details = {
                'predicted_class': predicted.item(),
                'confidence': confidence_score,
                'probabilities': {
                    'normal': probabilities[0][0].item(),
                    'ddos': probabilities[0][1].item()
                }
            }
            
            return prediction_label, confidence_score, details
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Return dummy prediction for demo
            is_attack = random.choice([True, False])
            if is_attack:
                return "DDoS Attack", random.uniform(0.7, 0.95), {
                    'predicted_class': 1,
                    'confidence': random.uniform(0.7, 0.95),
                    'probabilities': {'normal': random.uniform(0.05, 0.3), 'ddos': random.uniform(0.7, 0.95)}
                }
            else:
                return "Normal Traffic", random.uniform(0.6, 0.9), {
                    'predicted_class': 0,
                    'confidence': random.uniform(0.6, 0.9),
                    'probabilities': {'normal': random.uniform(0.6, 0.9), 'ddos': random.uniform(0.1, 0.4)}
                }

# Khởi tạo detector
detector = DDoSDetector()

# Biến global để lưu trạng thái
current_stats = {
    'threat_level': 0.3,
    'packets_per_sec': 10,
    'bytes_per_sec': 6226,
    'unique_ips': 9,
    'attack_detected': False,
    'last_prediction': 'Normal Traffic',
    'confidence': 0.0
}

def simulate_network_traffic():
    """Simulate network traffic data"""
    return {
        'packet_length': random.randint(64, 1500),
        'protocol': random.choice([6, 17, 1]),
        'source_port': random.randint(1024, 65535),
        'dest_port': random.choice([80, 443, 53, 22, 21]),
        'packet_rate': random.uniform(1, 1000),
        'byte_rate': random.uniform(100, 10000),
        'flow_duration': random.uniform(0.1, 60.0),
        'packet_count': random.randint(1, 100),
        'unique_sources': random.randint(1, 50),
        'unique_destinations': random.randint(1, 10)
    }

def update_stats_continuously():
    """Background thread để cập nhật stats liên tục"""
    global current_stats
    while True:
        try:
            # Simulate network traffic
            traffic_data = simulate_network_traffic()
            
            # Predict using model
            prediction, confidence, details = detector.predict(traffic_data)
            
            # Update stats
            current_stats.update({
                'threat_level': details['probabilities']['ddos'] * 100,
                'packets_per_sec': random.randint(5, 100),
                'bytes_per_sec': random.randint(1000, 20000),
                'unique_ips': random.randint(5, 50),
                'attack_detected': prediction == "DDoS Attack",
                'last_prediction': prediction,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
            
            time.sleep(2)  # Update mỗi 2 giây
            
        except Exception as e:
            logger.error(f"Error in stats update: {e}")
            time.sleep(5)

# Start background thread
stats_thread = threading.Thread(target=update_stats_continuously, daemon=True)
stats_thread.start()

@app.route('/')
def index():
    """Serve HTML page"""
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """API để lấy stats hiện tại"""
    return jsonify(current_stats)

@app.route('/api/predict', methods=['POST'])
def predict_ddos():
    """API để predict DDoS từ data được gửi lên"""
    try:
        data = request.json
        prediction, confidence, details = detector.predict(data)
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'threat_level': details['probabilities']['ddos'] * 100,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/traffic')
def get_traffic_data():
    """API để lấy traffic data cho chart"""
    # Generate dummy traffic data
    traffic_data = []
    for i in range(10):
        traffic_data.append({
            'time': (datetime.now().timestamp() - (9-i) * 2) * 1000,  # milliseconds
            'value': random.randint(1, 12)
        })
    
    return jsonify(traffic_data)

@app.route('/api/simulate_attack', methods=['POST'])
def simulate_attack():
    """API để simulate attack"""
    global current_stats
    attack_type = request.json.get('type', 'ddos')
    
    if attack_type == 'stop':
        current_stats['attack_detected'] = False
        current_stats['last_prediction'] = 'Normal Traffic'
        current_stats['threat_level'] = random.uniform(0.1, 2.0)
    else:
        current_stats['attack_detected'] = True
        current_stats['last_prediction'] = 'DDoS Attack'
        current_stats['threat_level'] = random.uniform(70, 95)
    
    return jsonify({'status': 'success', 'attack_type': attack_type})

if __name__ == '__main__':
    print("Starting DDoS Detection Server...")
    print("Access the dashboard at: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)