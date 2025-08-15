from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import random
import time
import threading
import json
from datetime import datetime
import logging
import pandas as pd
from collections import defaultdict, deque
import ipaddress

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

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
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class DDoSIPTracker:
    def __init__(self, window_size=100, threshold_rate=50, threshold_confidence=0.7):
        self.window_size = window_size
        self.threshold_rate = threshold_rate
        self.threshold_confidence = threshold_confidence
        
        self.ip_packet_count = defaultdict(int)
        self.ip_byte_count = defaultdict(int)
        self.ip_timestamps = defaultdict(lambda: deque(maxlen=window_size))
        self.ip_predictions = defaultdict(list)
        
        self.ddos_ips = set()
        self.suspicious_ips = set()
        
        self.ip_details = defaultdict(lambda: {
            'first_seen': None,
            'last_seen': None,
            'total_packets': 0,
            'total_bytes': 0,
            'attack_probability': 0.0,
            'attack_count': 0,
            'normal_count': 0,
            'attack_types': defaultdict(int)
        })
    
    def generate_random_ip(self):
        return f"{random.randint(1, 223)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
    
    def update_ip_activity(self, ip, packet_size, prediction, confidence, attack_type=None):
        current_time = datetime.now()
        
        self.ip_packet_count[ip] += 1
        self.ip_byte_count[ip] += packet_size
        self.ip_timestamps[ip].append(current_time)
        self.ip_predictions[ip].append((prediction, confidence, current_time))
        
        details = self.ip_details[ip]
        if details['first_seen'] is None:
            details['first_seen'] = current_time
        details['last_seen'] = current_time
        details['total_packets'] += 1
        details['total_bytes'] += packet_size
        
        if prediction == "DDoS Attack":
            details['attack_count'] += 1
            if attack_type:
                details['attack_types'][attack_type] += 1
        else:
            details['normal_count'] += 1
        
        total_predictions = details['attack_count'] + details['normal_count']
        details['attack_probability'] = details['attack_count'] / total_predictions if total_predictions > 0 else 0
        
        self._classify_ip(ip)
    
    def _classify_ip(self, ip):
        details = self.ip_details[ip]
        timestamps = self.ip_timestamps[ip]
        
        if len(timestamps) >= 10:
            time_window = (timestamps[-1] - timestamps[0]).total_seconds()
            if time_window > 0:
                packet_rate = len(timestamps) / time_window
                if packet_rate > self.threshold_rate:
                    self.suspicious_ips.add(ip)
        
        if details['attack_probability'] > self.threshold_confidence:
            self.ddos_ips.add(ip)
            if ip in self.suspicious_ips:
                self.suspicious_ips.remove(ip)
        
        recent_predictions = [p for p in self.ip_predictions[ip] 
                            if (datetime.now() - p[2]).total_seconds() < 30]
        if len(recent_predictions) >= 5:
            attack_predictions = [p for p in recent_predictions if p[0] == "DDoS Attack"]
            if len(attack_predictions) / len(recent_predictions) > 0.6:
                self.ddos_ips.add(ip)
    
    def get_ddos_ips(self):
        ddos_list = []
        for ip in self.ddos_ips:
            details = self.ip_details[ip]
            ddos_list.append({
                'ip': ip,
                'attack_probability': round(details['attack_probability'], 3),
                'total_packets': details['total_packets'],
                'total_bytes': details['total_bytes'],
                'first_seen': details['first_seen'].isoformat() if details['first_seen'] else None,
                'last_seen': details['last_seen'].isoformat() if details['last_seen'] else None,
                'duration': (details['last_seen'] - details['first_seen']).total_seconds() 
                           if details['first_seen'] and details['last_seen'] else 0,
                'attack_types': dict(details['attack_types'])
            })
        return sorted(ddos_list, key=lambda x: x['attack_probability'], reverse=True)
    
    def get_suspicious_ips(self):
        suspicious_list = []
        for ip in self.suspicious_ips:
            details = self.ip_details[ip]
            timestamps = self.ip_timestamps[ip]
            
            packet_rate = 0
            if len(timestamps) >= 2:
                time_window = (timestamps[-1] - timestamps[0]).total_seconds()
                if time_window > 0:
                    packet_rate = len(timestamps) / time_window
            
            suspicious_list.append({
                'ip': ip,
                'packet_rate': round(packet_rate, 2),
                'total_packets': details['total_packets'],
                'attack_probability': round(details['attack_probability'], 3),
                'attack_types': dict(details['attack_types'])
            })
        return sorted(suspicious_list, key=lambda x: x['packet_rate'], reverse=True)
    
    def clear_detected_ips(self):
        self.ddos_ips.clear()
        self.suspicious_ips.clear()
        self.ip_packet_count.clear()
        self.ip_byte_count.clear()
        self.ip_timestamps.clear()
        self.ip_predictions.clear()
        self.ip_details.clear()

class DDoSDetector:
    def __init__(self, model_path='../best_model.pth', sequence_length=10):
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_buffer = []
        
        self.ip_tracker = DDoSIPTracker()
        
        try:
            self.load_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"Model file not found: {e}. Using dummy model for demo.")
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
        
        features = np.array(features, dtype=np.float32)
        if np.std(features) > 1e-8:
            features = (features - np.mean(features)) / np.std(features)
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
    
    def predict_with_ip_tracking(self, raw_data, source_ip=None, attack_type=None):
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
            
            if source_ip is None:
                source_ip = self.ip_tracker.generate_random_ip()
            
            packet_size = raw_data.get('packet_length', 0)
            self.ip_tracker.update_ip_activity(source_ip, packet_size, prediction_label, confidence_score, attack_type)
            
            details = {
                'predicted_class': predicted.item(),
                'confidence': confidence_score,
                'source_ip': source_ip,
                'probabilities': {
                    'normal': probabilities[0][0].item(),
                    'ddos': probabilities[0][1].item()
                },
                'attack_type': attack_type
            }
            
            return prediction_label, confidence_score, details
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            source_ip = source_ip or self.ip_tracker.generate_random_ip()
            is_attack = random.choice([True, False])
            
            if is_attack:
                prediction = "DDoS Attack"
                confidence = random.uniform(0.7, 0.95)
                probs = {'normal': random.uniform(0.05, 0.3), 'ddos': random.uniform(0.7, 0.95)}
            else:
                prediction = "Normal Traffic"  
                confidence = random.uniform(0.6, 0.9)
                probs = {'normal': random.uniform(0.6, 0.9), 'ddos': random.uniform(0.1, 0.4)}
            
            packet_size = raw_data.get('packet_length', random.randint(64, 1500))
            self.ip_tracker.update_ip_activity(source_ip, packet_size, prediction, confidence, attack_type)
            
            return prediction, confidence, {
                'predicted_class': 1 if is_attack else 0,
                'confidence': confidence,
                'source_ip': source_ip,
                'probabilities': probs,
                'attack_type': attack_type
            }

# Global variables
detector = DDoSDetector()
model = None
scaler = None  
label_encoder = None

current_stats = {
    'threat_level': 0.3,
    'packets_per_sec': 10,
    'bytes_per_sec': 6226,
    'unique_ips': 9,
    'attack_detected': False,
    'last_prediction': 'Normal Traffic',
    'confidence': 0.0,
    'ddos_ip_count': 0,
    'suspicious_ip_count': 0,
    'attack_type': None
}

# --- Traffic history buffer for real chart data ---
traffic_history = []
TRAFFIC_HISTORY_SIZE = 30

# --- Traffic history buffer for real chart data ---
traffic_history = []
TRAFFIC_HISTORY_SIZE = 30

def simulate_network_traffic_with_ip(attack_type=None):
    ip_types = ['normal', 'suspicious', 'attack']
    ip_type = random.choices(ip_types, weights=[0.7, 0.2, 0.1])[0] if attack_type is None else 'attack'
    
    if ip_type == 'attack':
        if attack_type == 'udp_flood':
            return {
                'packet_length': random.randint(64, 200),
                'protocol': 17,  # UDP
                'source_port': random.randint(1024, 65535),
                'dest_port': random.choice([53, 123, 161]),  # Common UDP ports
                'packet_rate': random.uniform(200, 1500),
                'byte_rate': random.uniform(10000, 100000),
                'flow_duration': random.uniform(0.05, 1.0),
                'packet_count': random.randint(100, 500),
                'unique_sources': random.randint(1, 3),
                'unique_destinations': 1,
                'source_ip': random.choice([
                    '192.168.1.100', '10.0.0.50', '172.16.1.200',
                    '203.0.113.10', '198.51.100.25'
                ]),
                'attack_type': 'udp_flood'
            }
        elif attack_type == 'http_flood':
            return {
                'packet_length': random.randint(200, 800),
                'protocol': 6,  # TCP
                'source_port': random.randint(1024, 65535),
                'dest_port': 80,  # HTTP
                'packet_rate': random.uniform(100, 800),
                'byte_rate': random.uniform(5000, 50000),
                'flow_duration': random.uniform(0.1, 2.0),
                'packet_count': random.randint(50, 200),
                'unique_sources': random.randint(5, 20),
                'unique_destinations': 1,
                'source_ip': random.choice([
                    '192.168.1.100', '10.0.0.50', '172.16.1.200',
                    '203.0.113.10', '198.51.100.25'
                ]),
                'attack_type': 'http_flood'
            }
    elif ip_type == 'suspicious':
        return {
            'packet_length': random.randint(200, 800),
            'protocol': random.choice([6, 17, 1]),
            'source_port': random.randint(1024, 65535),
            'dest_port': random.choice([80, 443, 53, 22, 21]),
            'packet_rate': random.uniform(20, 80),
            'byte_rate': random.uniform(1000, 10000),
            'flow_duration': random.uniform(1.0, 10.0),
            'packet_count': random.randint(10, 50),
            'unique_sources': random.randint(1, 20),
            'unique_destinations': random.randint(1, 10),
            'source_ip': f"192.168.{random.randint(1, 255)}.{random.randint(1, 254)}",
            'attack_type': None
        }
    else:
        return {
            'packet_length': random.randint(64, 1500),
            'protocol': random.choice([6, 17, 1]),
            'source_port': random.randint(1024, 65535),
            'dest_port': random.choice([80, 443, 53, 22, 21]),
            'packet_rate': random.uniform(1, 20),
            'byte_rate': random.uniform(100, 2000),
            'flow_duration': random.uniform(5.0, 60.0),
            'packet_count': random.randint(1, 20),
            'unique_sources': random.randint(1, 50),
            'unique_destinations': random.randint(1, 10),
            'source_ip': f"10.0.{random.randint(1, 255)}.{random.randint(1, 254)}",
            'attack_type': None
        }

def update_stats_continuously():
    global current_stats
    while True:
        try:
            attack_type = current_stats.get('attack_type')
            traffic_data = simulate_network_traffic_with_ip(attack_type)
            
            source_ip = traffic_data.get('source_ip')
            attack_type = traffic_data.get('attack_type')
            prediction, confidence, details = detector.predict_with_ip_tracking(traffic_data, source_ip, attack_type)
            
            ddos_ips = detector.ip_tracker.get_ddos_ips()
            suspicious_ips = detector.ip_tracker.get_suspicious_ips()
            
            current_stats.update({
                'threat_level': details['probabilities']['ddos'] * 100,
                'packets_per_sec': traffic_data['packet_rate'],
                'bytes_per_sec': traffic_data['byte_rate'],
                'unique_ips': traffic_data['unique_sources'],
                'attack_detected': prediction == "DDoS Attack",
                'last_prediction': prediction,
                'confidence': confidence,
                'ddos_ip_count': len(ddos_ips),
                'suspicious_ip_count': len(suspicious_ips),
                'attack_type': attack_type,
                'timestamp': datetime.now().isoformat()
            })
            
            # --- Update traffic history buffer ---
            traffic_history.append({
                'time': datetime.now().timestamp() * 1000,
                'packets': current_stats['packets_per_sec'],
                'bytes': current_stats['bytes_per_sec'],
                'threat_level': current_stats['threat_level']
            })
            if len(traffic_history) > TRAFFIC_HISTORY_SIZE:
                traffic_history.pop(0)
            
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error in stats update: {e}")
            time.sleep(5)

# Start background thread
stats_thread = threading.Thread(target=update_stats_continuously, daemon=True)
stats_thread.start()

def load_model_and_preprocessors(load_path='../best_modal.pth'):
    global model, scaler, label_encoder
    try:
        checkpoint = torch.load(load_path, map_location='cpu')
        model_type = checkpoint.get('model_type', 'CNN_LSTM')
        arch = checkpoint.get('model_architecture', {})
        
        if model_type == 'CNN_LSTM':
            model = CNNLSTMModel(
                input_features=arch.get('input_size', 10),
                sequence_length=arch.get('sequence_length', 10),
                num_classes=arch.get('num_classes', 2)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        scaler = checkpoint.get('scaler')
        label_encoder = checkpoint.get('label_encoder')
        
        logger.info("Model and preprocessors loaded successfully")
        return model, scaler, label_encoder
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None, None

try:
    csv_data_df = pd.read_csv('combined_data.csv')
    logger.info(f"CSV data loaded: {len(csv_data_df)} rows")
except Exception as e:
    logger.warning(f"Could not load CSV data: {e}")
    csv_data_df = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    return jsonify(current_stats)

@app.route('/api/predict', methods=['POST'])
def predict_ddos():
    try:
        data = request.json
        source_ip = data.get('source_ip')
        attack_type = data.get('attack_type')
        prediction, confidence, details = detector.predict_with_ip_tracking(data, source_ip, attack_type)
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'threat_level': details['probabilities']['ddos'] * 100,
            'source_ip': details['source_ip'],
            'attack_type': attack_type,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/traffic')
def get_traffic_data():
    return jsonify(traffic_history)

@app.route('/api/ddos_ips', methods=['GET'])
def get_ddos_ips():
    ddos_ips = detector.ip_tracker.get_ddos_ips()
    suspicious_ips = detector.ip_tracker.get_suspicious_ips()
    
    return jsonify({
        'ddos_ips': ddos_ips,
        'suspicious_ips': suspicious_ips,
        'total_ddos': len(ddos_ips),
        'total_suspicious': len(suspicious_ips)
    })

@app.route('/api/sample', methods=['GET'])
def get_sample_and_predict():
    global model, scaler, label_encoder, csv_data_df
    
    if csv_data_df is None:
        return jsonify({'error': 'CSV data not available'}), 404
    
    try:
        if model is None:
            model, scaler, label_encoder = load_model_and_preprocessors()
            if model is None:
                return jsonify({'error': 'Model not available'}), 500
        
        sample = csv_data_df.sample(1).iloc[0]
        features = sample.drop('Label') if 'Label' in sample else sample
        true_label = sample.get('Label', 'Unknown')
        
        source_ip = sample.get('source_ip', sample.get('Source IP', None))
        attack_type = sample.get('attack_type', None)
        
        features_arr = features.values.astype(float).reshape(1, -1)
        if scaler is not None:
            features_scaled = scaler.transform(features_arr)
        else:
            features_scaled = features_arr
            
        input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()
            
        if label_encoder is not None:
            pred_label = label_encoder.classes_[pred_idx]
            prob_dict = {c: float(probs[0][i]) for i, c in enumerate(label_encoder.classes_)}
        else:
            pred_label = "DDoS Attack" if pred_idx == 1 else "Normal Traffic"
            prob_dict = {'Normal': float(probs[0][0]), 'DDoS': float(probs[0][1])}
        
        if source_ip and pred_label in ["DDoS Attack", "DDoS", "DDOS"]:
            packet_size = features.get('packet_length', features.get('Packet Length', 64))
            detector.ip_tracker.update_ip_activity(source_ip, packet_size, pred_label, confidence, attack_type)
        
        return jsonify({
            'features': features.to_dict(),
            'true_label': true_label,
            'predicted_label': pred_label,
            'confidence': confidence,
            'source_ip': source_ip,
            'attack_type': attack_type,
            'probabilities': prob_dict,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in sample prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulate_attack', methods=['POST'])
def simulate_attack():
    global current_stats
    attack_type = request.json.get('type', 'ddos')
    
    if attack_type == 'stop':
        current_stats['attack_detected'] = False
        current_stats['last_prediction'] = 'Normal Traffic'
        current_stats['threat_level'] = random.uniform(0.1, 2.0)
        current_stats['attack_type'] = None
        detector.ip_tracker.clear_detected_ips()
        current_stats['ddos_ip_count'] = 0
        current_stats['suspicious_ip_count'] = 0
    else:
        current_stats['attack_detected'] = True
        current_stats['last_prediction'] = 'DDoS Attack'
        current_stats['threat_level'] = random.uniform(70, 95)
        current_stats['attack_type'] = attack_type
        
        attack_ips = [
            '192.168.1.100', '10.0.0.50', '172.16.1.200',
            '203.0.113.10', '198.51.100.25', '192.168.1.150'
        ]
        
        for ip in attack_ips[:random.randint(3, 6)]:
            packet_size = random.randint(64, 200)
            detector.ip_tracker.update_ip_activity(ip, packet_size, "DDoS Attack", random.uniform(0.8, 0.95), attack_type)
        
        current_stats['ddos_ip_count'] = len(detector.ip_tracker.get_ddos_ips())
        current_stats['suspicious_ip_count'] = len(detector.ip_tracker.get_suspicious_ips())
    
    return jsonify({
        'status': 'success', 
        'attack_type': attack_type,
        'ddos_ip_count': current_stats['ddos_ip_count'],
        'suspicious_ip_count': current_stats['suspicious_ip_count']
    })

if __name__ == '__main__':
    print("Starting DDoS Detection Server...")
    print("Access the dashboard at: http://localhost:5001")
    print("API Endpoints:")
    print("  - GET  /api/stats           - Current statistics")
    print("  - GET  /api/ddos_ips        - List of detected DDoS IPs")
    print("  - POST /api/predict         - Predict DDoS from data")
    print("  - GET  /api/sample          - Get sample from CSV and predict")
    print("  - POST /api/simulate_attack - Simulate attack/stop")
    app.run(debug=True, host='0.0.0.0', port=5001)