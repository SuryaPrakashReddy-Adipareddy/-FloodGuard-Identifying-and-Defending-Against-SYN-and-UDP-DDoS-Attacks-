import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,roc_curve, auc, precision_recall_curve)
from imblearn.over_sampling import SMOTE
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=3, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = 1e-6

    def forward(self, inputs, targets):
        
        inputs_prob = torch.sigmoid(inputs)
        inputs_prob = torch.clamp(inputs_prob, self.epsilon, 1.0 - self.epsilon)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = targets * inputs_prob + (1 - targets) * (1 - inputs_prob)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_term = (1 - p_t) ** self.gamma
        loss = alpha_factor * focal_term * ce_loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def attention_net(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        attn_output = self.attention_net(lstm_out)
        out = self.fc_layers(attn_output)
        return out

def create_output_directory():
    output_dir = "ddos_analysis_output"
    plots_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return output_dir

def clean_dataframe(df):
    df_cleaned = df.dropna()
    df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan).dropna()
    numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    df_cleaned[numeric_columns] = df_cleaned[numeric_columns].astype(np.float32)
    return df_cleaned

def load_and_preprocess_data(syn_train_path, syn_test_path, udp_train_path, udp_test_path):
    print("\nLoading and preprocessing datasets...")
    
    try:
        syn_train = pd.read_csv(syn_train_path)
        syn_test = pd.read_csv(syn_test_path)
        udp_train = pd.read_csv(udp_train_path)
        udp_test = pd.read_csv(udp_test_path)
        
        syn_train = clean_dataframe(syn_train)
        syn_test = clean_dataframe(syn_test)
        udp_train = clean_dataframe(udp_train)
        udp_test = clean_dataframe(udp_test)
        
        train_df = pd.concat([syn_train, udp_train])
        test_df = pd.concat([syn_test, udp_test])
        
        label_map = {'Benign': 0, 'Syn': 1, 'UDP': 1}
        train_df['Label'] = train_df['Label'].map(label_map)
        test_df['Label'] = test_df['Label'].map(label_map)
        
        train_df = train_df.dropna(subset=['Label'])
        test_df = test_df.dropna(subset=['Label'])
        
        train_df['Label'] = train_df['Label'].astype(int)
        test_df['Label'] = test_df['Label'].astype(int)
        
        print(f"Training set shape: {train_df.shape}")
        print(f"Testing set shape: {test_df.shape}")
        
        return train_df, test_df
        
    except Exception as e:
        print(f"Error in data loading: {str(e)}")
        raise

def create_visualizations(df, output_dir, prefix=""):
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.select_dtypes(include=[np.number]).columns[:6]):
        plt.subplot(2, 3, i+1)
        sns.kdeplot(data=df, x=column, hue='Label', common_norm=False)
        plt.title(f'Distribution of {column}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", f"{prefix}feature_distributions.png"))
    plt.close()

    plt.figure(figsize=(12, 10))
    correlation = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", f"{prefix}correlation_heatmap.png"))
    plt.close()

def preprocess_data(train_df, test_df):
    print("\nPreprocessing data for model training...")
    
    try:
        X_train = train_df.drop('Label', axis=1)
        y_train = train_df['Label']
        X_test = test_df.drop('Label', axis=1)
        y_test = test_df['Label']
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_scaled = X_train_scaled.astype(np.float32)
        X_test_scaled = X_test_scaled.astype(np.float32)
        
        print("Applying SMOTE for improved class balancing...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        # Create class distribution visualization
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.countplot(x=y_train)
        plt.title('Class Distribution Before SMOTE')
        plt.xlabel('Class (0: Normal, 1: DDoS)')
        
        plt.subplot(1, 2, 2)
        sns.countplot(x=y_train_resampled)
        plt.title('Class Distribution After SMOTE')
        plt.xlabel('Class (0: Normal, 1: DDoS)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", "class_distribution.png"))
        plt.close()
        
        X_train_lstm = X_train_resampled.reshape(X_train_resampled.shape[0], 1, X_train_resampled.shape[1])
        X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
        
        return X_train_lstm, X_test_lstm, y_train_resampled, y_test
        
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        raise

def train_model(X_train, y_train, model, output_dir, num_epochs=20, batch_size=64, learning_rate=0.001):
    print("\nTraining enhanced LSTM model with Focal Loss...")
    
    criterion = FocalLoss(alpha=0.25, gamma=3)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    epochs = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        epochs.append(epoch)
        
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "plots", "training_loss.png"))
    plt.close()
    
    return losses

def evaluate_model(model, X_test, y_test, output_dir):
    print("\nEvaluating model...")
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        outputs = model(X_test_tensor)
        predicted = (torch.sigmoid(outputs) > 0.5).int().squeeze()
        y_pred = predicted.numpy()
        y_prob = torch.sigmoid(outputs).numpy().squeeze()

    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'DDoS'],
                yticklabels=['Normal', 'DDoS'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    for i in range(2):
        for j in range(2):
            plt.text(j + 0.5, i + 0.5, f'{(cm[i, j]/cm.sum())*100:.1f}%',
                    ha='center', va='center')
    plt.savefig(os.path.join(output_dir, "plots", "confusion_matrix.png"))
    plt.close()

    # ROC and Precision-Recall Curves
    plt.figure(figsize=(15, 6))
    
    # ROC Curve
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, color='green', lw=2,label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "roc_pr_curves.png"))
    plt.close()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save metrics to file
    with open(os.path.join(output_dir, "model_metrics.txt"), "w") as f:
        f.write("Model Evaluation Metrics\n")
        f.write("=======================\n\n")
        f.write(f"ROC AUC Score: {roc_auc:.3f}\n")
        f.write(f"PR AUC Score: {pr_auc:.3f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))

if __name__ == "__main__":
    print("Starting DDoS detection script...")
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory created: {output_dir}")
    
    # Update these paths to match your file names
    syn_train_path = "Syn-training.csv"
    syn_test_path = "Syn-testing.csv"
    udp_train_path = "UDP-training.csv"
    udp_test_path = "UDP-testing.csv"
    
    try:
        # Load and preprocess data
        train_df, test_df = load_and_preprocess_data(syn_train_path, syn_test_path, 
                                                    udp_train_path, udp_test_path)
        print("Data loaded and preprocessed.")
        
        # Create initial visualizations
        create_visualizations(train_df, output_dir, prefix="train_")
        create_visualizations(test_df, output_dir, prefix="test_")
        print("Initial visualizations created.")
        
        # Prepare data for model
        X_train_lstm, X_test_lstm, y_train_resampled, y_test = preprocess_data(train_df, test_df)
        print("Data preprocessed for model training.")
        
        # Initialize model
        input_size = X_train_lstm.shape[2]
        hidden_size = 64
        num_layers = 2
        num_classes = 1
        
        model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
        print("Model initialized.")
        
        # Train model
        losses = train_model(X_train_lstm, y_train_resampled, model, output_dir)
        print("Model training completed.")
        
        # Evaluate model and create final visualizations
        evaluate_model(model, X_test_lstm, y_test, output_dir)
        print("Model evaluation completed.")
        

        # Save model
        model_save_path = os.path.join(output_dir, "ddos_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_classes': num_classes
        }, model_save_path)
        print(f"Model saved to {model_save_path}")
        
        print("\nGenerated visualizations:")
        print("1. Feature distributions (train_feature_distributions.png)")
        print("2. Correlation heatmap (train_correlation_heatmap.png)")
        print("3. Class distribution before/after SMOTE (class_distribution.png)")
        print("4. Training loss curve (training_loss.png)")
        print("5. Confusion matrix (confusion_matrix.png)")
        print("6. ROC and PR curves (roc_pr_curves.png)")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise
    
    print("\nDDoS detection script finished successfully.")
    print(f"All outputs have been saved to: {output_dir}")