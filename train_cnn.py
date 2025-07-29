#!/usr/bin/env python3
"""
CIC-DDoS2019 CNN Training Script
Chuyển đổi từ Jupyter notebook để training CNN model phát hiện DDoS attacks
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

class CNN(nn.Module):
    """
    CNN Model cho phát hiện DDoS attacks
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
    
    # Tìm cột label (có thể là 'Label' hoặc 'Label')
    label_col = None
    for col in ['Label', 'Label']:
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
                num_epochs=200, learning_rate=0.001):
    """
    Training CNN model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_acc = []
    test_acc = []
    train_losses = []
    
    print(f"Bắt đầu training với {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        correct_train = 0
        total_train = 0
        epoch_loss = 0
        
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
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
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss: {train_losses[-1]:.4f}, '
                  f'Train Acc: {train_accuracy:.2f}%, '
                  f'Test Acc: {test_accuracy:.2f}%')
    
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

def save_model(model, scaler, label_encoder, save_path='cnn_ddos_model.pth'):
    """
    Lưu model và preprocessors
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'label_encoder': label_encoder,
        'model_architecture': {
            'input_size': model.fc.in_features // 32,  # Reverse calculate
            'num_classes': model.fc.out_features
        }
    }, save_path)
    print(f"Model đã được lưu tại: {save_path}")

def main():
    """
    Main function để chạy training
    """
    print("=== CIC-DDoS2019 CNN Training ===")
    
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
    model = CNN(input_size, num_classes)
    
    print(f"Model architecture:")
    print(f"- Input size: {input_size}")
    print(f"- Number of classes: {num_classes}")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 4. Training
    train_acc, test_acc, train_losses = train_model(
        model, train_loader, test_loader, X_test_tensor, y_test_tensor,
        num_epochs=200, learning_rate=0.001
    )
    
    # 5. Evaluation
    all_preds, all_labels = evaluate_model(
        model, X_test_tensor, y_test_tensor, label_encoder
    )
    
    # 6. Plot results
    plot_results(train_acc, test_acc, train_losses, all_labels, all_preds)
    
    # 7. Save model
    save_model(model, scaler, label_encoder)
    
    print("\nTraining hoàn thành!")

if __name__ == "__main__":
    # Thêm option để chỉ kiểm tra dữ liệu
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--inspect":
        inspect_data('./combined_data.csv')
    else:
        main()