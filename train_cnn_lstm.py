#!/usr/bin/env python3
"""
CIC-DIAD 2024 CNN-LSTM Training Script
Training CNN-LSTM model phát hiện DDoS attacks
Hỗ trợ cả CNN và CNN-LSTM architectures với attention mechanism
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CNN_LSTM(nn.Module):
    """
    CNN-LSTM Model cho phát hiện DDoS attacks
    Kết hợp CNN để trích xuất đặc trưng và LSTM để học mẫu tuần tự
    """
    def __init__(self, input_size, num_classes, cnn_channels=[16, 32, 64], 
                 lstm_hidden_size=128, lstm_num_layers=2, dropout=0.3):
        super(CNN_LSTM, self).__init__()
        
        self.input_size = input_size
        self.cnn_channels = cnn_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        
        # CNN layers để trích xuất đặc trưng
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        in_channels = 1
        for out_channels in cnn_channels:
            self.conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            self.bn_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
        
        # Tính toán kích thước sau các CNN layers
        conv_output_size = input_size
        for _ in cnn_channels:
            conv_output_size = conv_output_size // 2  # Sau mỗi pooling layer
        
        # LSTM layers để học mẫu tuần tự
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],  # Số channels cuối cùng của CNN
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True  # Sử dụng bidirectional LSTM
        )
        
        # Attention mechanism
        self.attention = nn.Linear(lstm_hidden_size * 2, 1)  # *2 vì bidirectional
        
        # Classification layers
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size // 2, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Thêm channel dimension cho CNN
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        # CNN feature extraction
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.dropout(x)
        
        # Chuyển đổi cho LSTM: (batch_size, channels, seq_len) -> (batch_size, seq_len, channels)
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.fc_layers(attended_output)
        
        return output

class CNN(nn.Module):
    """
    CNN Model đơn giản (giữ lại để backward compatibility)
    """
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Tính toán kích thước sau pooling
        conv1_out_size = (input_size + 2 * 1 - 3) / 1 + 1  # Conv1
        pool1_out_size = conv1_out_size / 2  # Pool1
        conv2_out_size = (pool1_out_size + 2 * 1 - 3) / 1 + 1  # Conv2
        pool2_out_size = conv2_out_size / 2  # Pool2
        final_size = int(pool2_out_size) * 32  # conv2 output channels * output length
        
        self.fc = nn.Linear(final_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Thêm channel dimension
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def inspect_data(data_path='./combined_data.csv', show_sample=True):
    """
    Kiểm tra cấu trúc dữ liệu trước khi xử lý
    """
    print("=== KIỂM TRA DỮ LIỆU ===")
    data = pd.read_csv(data_path)
    
    print(f"Kích thước dữ liệu: {data.shape}")
    print(f"Tổng số cột: {len(data.columns)}")
    
    print("\nCác cột trong dữ liệu:")
    for i, col in enumerate(data.columns):
        print(f"{i+1:2d}. '{col}'")
    
    # Kiểm tra các loại dữ liệu
    print(f"\nKiểm tra data types:")
    print(data.dtypes.value_counts())
    
    # Kiểm tra missing values
    print(f"\nMissing values:")
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("Không có missing values")
    
    # Kiểm tra infinite values
    print(f"\nKiểm tra infinite values...")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    inf_count = 0
    for col in numeric_cols:
        inf_in_col = np.isinf(data[col]).sum()
        if inf_in_col > 0:
            print(f"  {col}: {inf_in_col} infinite values")
            inf_count += inf_in_col
    
    if inf_count == 0:
        print("Không có infinite values trong các cột numeric")
    
    # Tìm cột label
    label_candidates = [col for col in data.columns if 'label' in col.lower()]
    print(f"\nCác cột có thể là label: {label_candidates}")
    
    if label_candidates:
        label_col = label_candidates[0]
        print(f"Phân bố các class trong '{label_col}':")
        print(data[label_col].value_counts())
    
    if show_sample and len(data) > 0:
        print(f"\n5 dòng đầu tiên:")
        print(data.head())
    
    return data

def load_and_preprocess_data(data_path='./combined_data.csv', inspect_first=False):
    """
    Load và tiền xử lý dữ liệu
    """
    if inspect_first:
        inspect_data(data_path)
        print("\n" + "="*50 + "\n")
    
    print("Đang load dữ liệu...")
    data = pd.read_csv(data_path)
    
    print(f"Kích thước dữ liệu gốc: {data.shape}")
    print(f"Các cột trong dữ liệu: {list(data.columns)}")
    
    # Định nghĩa các cột cần loại bỏ (không phải numeric features)
    columns_to_drop = [
        'Flow ID',           # ID của flow
        'Src IP',           # Source IP
        'Src Port',         # Source Port  
        'Dst IP',           # Destination IP
        'Dst Port',         # Destination Port
        'Protocol',         # Protocol number
        'Timestamp',        # Timestamp
        ' Timestamp'        # Timestamp với space
    ]
    
    # Loại bỏ các cột không cần thiết nếu tồn tại
    existing_cols_to_drop = [col for col in columns_to_drop if col in data.columns]
    if existing_cols_to_drop:
        print(f"Loại bỏ các cột: {existing_cols_to_drop}")
        data.drop(existing_cols_to_drop, axis=1, inplace=True)
    
    # Tìm cột label
    label_col = None
    for col in ['Label']:
        if col in data.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError("Không tìm thấy cột Label trong dữ liệu!")
    
    print(f"Sử dụng cột label: '{label_col}'")
    
    # Encode labels
    label_encoder = LabelEncoder()
    data[label_col] = label_encoder.fit_transform(data[label_col])
    
    print(f"Các class labels: {label_encoder.classes_}")
    
    # Tách features và labels
    X_data = data.drop(label_col, axis=1)
    y_data = data[label_col]
    
    print(f"Số features sau khi loại bỏ: {X_data.shape[1]}")
    
    # Chuyển đổi tất cả các cột còn lại thành numeric, xử lý lỗi
    print("Chuyển đổi dữ liệu sang numeric...")
    for col in X_data.columns:
        X_data[col] = pd.to_numeric(X_data[col], errors='coerce')
    
    # Kiểm tra và báo cáo các cột có vấn đề
    problematic_cols = []
    for col in X_data.columns:
        if X_data[col].isna().sum() > len(X_data) * 0.5:  # Nếu >50% là NaN
            problematic_cols.append(col)
    
    if problematic_cols:
        print(f"Cảnh báo: Các cột có nhiều NaN values: {problematic_cols}")
    
    # Xử lý inf values và NaN
    print("Xử lý inf values và NaN...")
    X_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Tính median chỉ cho các cột numeric
    numeric_cols = X_data.select_dtypes(include=[np.number]).columns
    print(f"Số cột numeric: {len(numeric_cols)}")
    
    # Fill NaN bằng median cho từng cột numeric
    for col in numeric_cols:
        if X_data[col].isna().sum() > 0:
            median_val = X_data[col].median()
            if pd.isna(median_val):  # Nếu median cũng là NaN, dùng 0
                median_val = 0
            X_data[col].fillna(median_val, inplace=True)
    
    # Loại bỏ các cột không phải numeric (nếu còn)
    X_data = X_data.select_dtypes(include=[np.number])
    
    print(f"Số features cuối cùng: {X_data.shape[1]}")
    
    # Chuẩn hóa features
    print("Chuẩn hóa features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X_data)
    y = y_data
    
    # Chia train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("Hoàn thành tiền xử lý dữ liệu!")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Class distribution in train set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for i, (class_idx, count) in enumerate(zip(unique, counts)):
        class_name = label_encoder.classes_[class_idx]
        print(f"  {class_name}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test, label_encoder, scaler

def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=64):
    """
    Tạo PyTorch data loaders
    """
    # Chuyển đổi sang torch tensors
    X_train_tensor = torch.tensor(X_train.astype(np.float32))
    y_train_tensor = torch.tensor(y_train.values.astype(np.int64))
    X_test_tensor = torch.tensor(X_test.astype(np.float32))
    y_test_tensor = torch.tensor(y_test.values.astype(np.int64))
    
    # Tạo datasets và loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, X_test_tensor, y_test_tensor

def train_model(model, train_loader, test_loader, X_test_tensor, y_test_tensor, 
                num_epochs=200, learning_rate=0.001, patience=20, model_type="CNN-LSTM"):
    """
    Training CNN-LSTM model với early stopping và learning rate scheduling
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    train_acc = []
    test_acc = []
    train_losses = []
    best_test_acc = 0
    patience_counter = 0
    
    print(f"Bắt đầu training {model_type} với {num_epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        correct_train = 0
        total_train = 0
        epoch_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping để tránh exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
        
        # Calculate training accuracy
        train_accuracy = 100 * correct_train / total_train
        train_acc.append(train_accuracy)
        train_losses.append(epoch_loss / len(train_loader))
        
        # Testing phase
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        # Calculate test accuracy
        test_accuracy = 100 * correct_test / total_test
        test_acc.append(test_accuracy)
        
        # Learning rate scheduling
        scheduler.step(test_accuracy)
        
        # Early stopping
        if epoch + 1 > 150:
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                patience_counter = 0
                # Lưu best model
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
        else:
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                # Lưu best model
                torch.save(model.state_dict(), 'best_model.pth')
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss: {train_losses[-1]:.4f}, '
                  f'Train Acc: {train_accuracy:.2f}%, '
                  f'Test Acc: {test_accuracy:.2f}%, '
                  f'Best: {best_test_acc:.2f}%, '
                  f'LR: {current_lr:.6f}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping tại epoch {epoch+1}")
            print(f"Best test accuracy: {best_test_acc:.2f}%")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    print(f"Loaded best model với accuracy: {best_test_acc:.2f}%")
    
    return train_acc, test_acc, train_losses

def evaluate_model(model, X_test_tensor, y_test_tensor, label_encoder):
    """
    Đánh giá model và tạo confusion matrix
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        test_loader = DataLoader(
            TensorDataset(X_test_tensor, y_test_tensor), 
            batch_size=64
        )
        
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    class_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=class_names))
    
    return all_preds, all_labels

def plot_results(train_acc, test_acc, train_losses, all_labels, all_preds):
    """
    Vẽ biểu đồ kết quả
    """
    # Plot accuracy
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(train_acc)+1), train_acc, label='Train Accuracy')
    plt.plot(range(1, len(test_acc)+1), test_acc, label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot confusion matrix
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    plt.tight_layout()
    plt.show()

def save_model(model, scaler, label_encoder, save_path='cnn_lstm_ddos_model.pth'):
    """
    Lưu model và preprocessors
    """
    model_info = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'label_encoder': label_encoder,
        'model_type': type(model).__name__,
        'model_architecture': {}
    }
    
    # Lưu thông tin architecture tùy theo loại model
    if isinstance(model, CNN_LSTM):
        model_info['model_architecture'] = {
            'input_size': model.input_size,
            'num_classes': model.fc_layers[-1].out_features,
            'cnn_channels': model.cnn_channels,
            'lstm_hidden_size': model.lstm_hidden_size,
            'lstm_num_layers': model.lstm_num_layers
        }
    elif isinstance(model, CNN):
        model_info['model_architecture'] = {
            'input_size': model.fc.in_features // 32,  # Reverse calculate
            'num_classes': model.fc.out_features
        }
    
    torch.save(model_info, save_path)
    print(f"Model đã được lưu tại: {save_path}")

def load_model(load_path='cnn_lstm_ddos_model.pth'):
    """
    Load model đã lưu
    """
    checkpoint = torch.load(load_path)
    model_type = checkpoint['model_type']
    arch = checkpoint['model_architecture']
    
    if model_type == 'CNN_LSTM':
        model = CNN_LSTM(
            input_size=arch['input_size'],
            num_classes=arch['num_classes'],
            cnn_channels=arch['cnn_channels'],
            lstm_hidden_size=arch['lstm_hidden_size'],
            lstm_num_layers=arch['lstm_num_layers']
        )
    elif model_type == 'CNN':
        model = CNN(
            input_size=arch['input_size'],
            num_classes=arch['num_classes']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler = checkpoint['scaler']
    label_encoder = checkpoint['label_encoder']
    
    print(f"Loaded {model_type} model from {load_path}")
    return model, scaler, label_encoder

def main(model_type="CNN-LSTM", epochs=200, lr=0.001):
    """
    Main function để chạy training
    """
    print(f"=== CIC-DDoS2019 {model_type} Training ===")
    
    # 1. Load và preprocess data (với option kiểm tra dữ liệu trước)
    X_train, X_test, y_train, y_test, label_encoder, scaler = load_and_preprocess_data(
        inspect_first=True  # Set True để kiểm tra dữ liệu trước khi xử lý
    )
    
    # 2. Tạo data loaders
    train_loader, test_loader, X_test_tensor, y_test_tensor = create_data_loaders(
        X_train, X_test, y_train, y_test
    )
    
    # 3. Khởi tạo model
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    if model_type == "CNN-LSTM":
        model = CNN_LSTM(
            input_size=input_size,
            num_classes=num_classes,
            cnn_channels=[16, 32, 64],      # CNN channels
            lstm_hidden_size=128,           # LSTM hidden size
            lstm_num_layers=2,              # LSTM layers
            dropout=0.3                     # Dropout rate
        )
    elif model_type == "CNN":
        model = CNN(input_size, num_classes)
    else:
        raise ValueError("model_type phải là 'CNN' hoặc 'CNN-LSTM'")
    
    print(f"\n{model_type} Model architecture:")
    print(f"- Input size: {input_size}")
    print(f"- Number of classes: {num_classes}")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if model_type == "CNN-LSTM":
        print(f"- CNN channels: {model.cnn_channels}")
        print(f"- LSTM hidden size: {model.lstm_hidden_size}")
        print(f"- LSTM layers: {model.lstm_num_layers}")
    
    # 4. Training
    train_acc, test_acc, train_losses = train_model(
        model, train_loader, test_loader, X_test_tensor, y_test_tensor,
        num_epochs=epochs, learning_rate=lr, model_type=model_type
    )
    
    # 5. Evaluation
    all_preds, all_labels = evaluate_model(
        model, X_test_tensor, y_test_tensor, label_encoder
    )
    
    # 6. Plot results
    plot_results(train_acc, test_acc, train_losses, all_labels, all_preds)
    
    # 7. Save model
    save_model(model, scaler, label_encoder, 
               save_path=f'{model_type.lower()}_ddos_model.pth')
    
    print(f"\n{model_type} Training hoàn thành!")
    
    return model, train_acc, test_acc, all_preds, all_labels

def compare_models():
    """
    So sánh hiệu suất giữa CNN và CNN-LSTM
    """
    print("=== SO SÁNH MODELS CNN vs CNN-LSTM ===")
    
    # Load data một lần
    X_train, X_test, y_train, y_test, label_encoder, scaler = load_and_preprocess_data()
    train_loader, test_loader, X_test_tensor, y_test_tensor = create_data_loaders(
        X_train, X_test, y_train, y_test
    )
    
    results = {}
    
    for model_name in ["CNN", "CNN-LSTM"]:
        print(f"\n--- Training {model_name} ---")
        
        input_size = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        if model_name == "CNN-LSTM":
            model = CNN_LSTM(input_size, num_classes)
        else:
            model = CNN(input_size, num_classes)
        
        # Training với ít epochs hơn cho comparison
        train_acc, test_acc, train_losses = train_model(
            model, train_loader, test_loader, X_test_tensor, y_test_tensor,
            num_epochs=100, learning_rate=0.001, model_type=model_name
        )
        
        # Evaluation
        all_preds, all_labels = evaluate_model(
            model, X_test_tensor, y_test_tensor, label_encoder
        )
        
        results[model_name] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'final_test_acc': test_acc[-1],
            'best_test_acc': max(test_acc),
            'params': sum(p.numel() for p in model.parameters())
        }
    
    # So sánh kết quả
    print("\n=== KẾT QUẢ SO SÁNH ===")
    for name, result in results.items():
        print(f"{name}:")
        print(f"  - Parameters: {result['params']:,}")
        print(f"  - Final Test Acc: {result['final_test_acc']:.2f}%")
        print(f"  - Best Test Acc: {result['best_test_acc']:.2f}%")
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        plt.plot(result['train_acc'], label=f'{name} Train')
        plt.plot(result['test_acc'], label=f'{name} Test', linestyle='--')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    models = list(results.keys())
    final_accs = [results[name]['final_test_acc'] for name in models]
    best_accs = [results[name]['best_test_acc'] for name in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, final_accs, width, label='Final Test Acc')
    plt.bar(x + width/2, best_accs, width, label='Best Test Acc')
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.title('Final Performance Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--inspect":
            inspect_data('./combined_data.csv')
        elif sys.argv[1] == "--compare":
            compare_models()
        elif sys.argv[1] == "--cnn":
            main(model_type="CNN")
        elif sys.argv[1] == "--cnn-lstm":
            main(model_type="CNN-LSTM")
        else:
            print("Usage: python script.py [--inspect|--compare|--cnn|--cnn-lstm]")
    else:
        # Mặc định chạy CNN-LSTM
        main(model_type="CNN-LSTM")