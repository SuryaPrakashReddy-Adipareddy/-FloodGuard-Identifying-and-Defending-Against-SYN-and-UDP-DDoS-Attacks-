import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weight of the rare class (DDoS attacks)
        self.gamma = gamma  # Down-weighting factor for easy examples
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate binary cross-entropy loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Probability of the correct class
        # Apply focal loss formula
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# LSTM Model for DDoS detection
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def create_output_directory():
    """Create directory for saving outputs"""
    output_dir = "ddos_analysis_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "plots"))
    return output_dir

def clean_dataframe(df):
    """Clean individual dataframe"""
    # Remove rows with NaN values
    df_cleaned = df.dropna()
    
    # Remove infinite values
    df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Ensure all numeric columns are float32
    numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    df_cleaned[numeric_columns] = df_cleaned[numeric_columns].astype(np.float32)
    
    return df_cleaned

def load_and_preprocess_data(syn_train_path, syn_test_path, udp_train_path, udp_test_path):
    """Load and preprocess all datasets"""
    print("\nLoading and preprocessing datasets...")
    
    try:
        # Load datasets
        syn_train = pd.read_csv(syn_train_path)
        syn_test = pd.read_csv(syn_test_path)
        udp_train = pd.read_csv(udp_train_path)
        udp_test = pd.read_csv(udp_test_path)
        
        # Clean datasets
        syn_train = clean_dataframe(syn_train)
        syn_test = clean_dataframe(syn_test)
        udp_train = clean_dataframe(udp_train)
        udp_test = clean_dataframe(udp_test)
        
        # Combine training data
        train_df = pd.concat([syn_train, udp_train])
        test_df = pd.concat([syn_test, udp_test])
        
        # Convert labels: 0 for benign, 1 for attacks (both SYN and UDP)
        label_map = {'Benign': 0, 'Syn': 1, 'UDP': 1}
        train_df['Label'] = train_df['Label'].map(label_map)
        test_df['Label'] = test_df['Label'].map(label_map)
        
        # Drop any rows where label mapping failed
        train_df = train_df.dropna(subset=['Label'])
        test_df = test_df.dropna(subset=['Label'])
        
        # Convert Label to int
        train_df['Label'] = train_df['Label'].astype(int)
        test_df['Label'] = test_df['Label'].astype(int)
        
        print(f"Training set shape: {train_df.shape}")
        print(f"Testing set shape: {test_df.shape}")
        
        return train_df, test_df
        
    except Exception as e:
        print(f"Error in data loading: {str(e)}")
        raise

def create_visualizations(df, output_dir, prefix=""):
    """Create and save visualizations"""
    try:
        # Create distribution plots
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(df.select_dtypes(include=[np.number]).columns[:5]):
            plt.subplot(2, 3, i+1)
            sns.histplot(data=df, x=column, hue='Label')
            plt.title(f'Distribution of {column}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", f"{prefix}feature_distributions.png"))
        plt.close()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 8))
        correlation = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation, annot=False, cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", f"{prefix}correlation_heatmap.png"))
        plt.close()
        
    except Exception as e:
        print(f"Error in visualization creation: {str(e)}")
        raise

def preprocess_data(train_df, test_df):
    """Preprocess the data for model training"""
    print("\nPreprocessing data for model training...")
    
    try:
        # Separate features and labels
        X_train = train_df.drop('Label', axis=1)
        y_train = train_df['Label']
        X_test = test_df.drop('Label', axis=1)
        y_test = test_df['Label']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to float32
        X_train_scaled = X_train_scaled.astype(np.float32)
        X_test_scaled = X_test_scaled.astype(np.float32)
        
        print("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        # Reshape for LSTM (samples, time steps, features)
        X_train_lstm = X_train_resampled.reshape(X_train_resampled.shape[0], 1, X_train_resampled.shape[1])
        X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
        
        print(f"Final training set shape: {X_train_lstm.shape}")
        print(f"Final test set shape: {X_test_lstm.shape}")
        
        return X_train_lstm, X_test_lstm, y_train_resampled, y_test
        
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        raise

def train_model(X_train, y_train, model, num_epochs=10):
    """Train the LSTM model using Focal Loss"""
    print("\nTraining LSTM model with Focal Loss...")
    
    criterion = FocalLoss(alpha=1, gamma=2)  # Focal Loss for handling class imbalance
    optimizer = torch.optim.Adam(model.parameters())
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    
    losses = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return losses

def evaluate_model(model, X_test, y_test, output_dir):
    """Evaluate model and create visualization"""
    print("\nEvaluating model...")
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        outputs = model(X_test_tensor)
        predicted = (torch.sigmoid(outputs) > 0.5).int().squeeze()
        y_pred = predicted.numpy()
        
        # Calculate probabilities for ROC curve
        y_prob = torch.sigmoid(outputs).numpy().squeeze()
    
    # Print DDoS attack detection results
    for i in range(len(y_test)):
        if y_pred[i] == 1:
            print(f"Sample {i+1}: DDoS attack detected!")
        else:
            print(f"Sample {i+1}: No DDoS attack detected.")
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, "plots", "confusion_matrix.png"))
    plt.close()
    
    # Create ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "plots", "roc_curve.png"))
    plt.close()
    
    # Print classification report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

def create_training_plot(losses, output_dir):
    """Create and save training loss plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(output_dir, "plots", "training_loss.png"))
    plt.close()

def main():
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create output directory
        output_dir = create_output_directory()
        
        # File paths
        base_path = "/mnt/c/work_folder/python_files/pystart/ddos"
        syn_train_path = os.path.join(base_path, "Syn-training.csv")
        syn_test_path = os.path.join(base_path, "Syn-testing.csv")
        udp_train_path = os.path.join(base_path, "UDP-training.csv")
        udp_test_path = os.path.join(base_path, "UDP-testing.csv")
        
        # Load and preprocess data
        train_df, test_df = load_and_preprocess_data(syn_train_path, syn_test_path, 
                                                    udp_train_path, udp_test_path)
        
        # Create visualizations
        create_visualizations(train_df, output_dir, "train_")
        
        # Preprocess data for model
        X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)
        
        # Initialize model
        input_size = X_train.shape[2]
        hidden_size = 64
        num_layers = 2
        num_classes = 1  # Binary classification: DDoS or not
        model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
        
        # Train model
        losses = train_model(X_train, y_train, model, num_epochs=10)
        
        # Create training plot
        create_training_plot(losses, output_dir)
        
        # Save model
        model_path = os.path.join(output_dir, "lstm_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"\nModel saved to {model_path}")
        
        # Evaluate model
        evaluate_model(model, X_test, y_test, output_dir)
        
        print(f"\nAnalysis complete! Results saved in {output_dir}")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()


# So, we're looking at a confusion matrix here. It's basically a scorecard for how well our model is doing at detecting DDoS attacks. Let's break it down:

# The top-left box (1938): This is the number of times our model correctly said "No DDoS attack" when there really wasn't one. That's pretty good!
# The top-right box (478): Oops! This is where our model said "DDoS attack!" but it was actually just normal traffic. We call these false alarms.
# The bottom-left box (0): This is great! Our model never missed a real DDoS attack. It didn't say "All clear" when there was actually an attack happening.
# The bottom-right box (533): This is the number of times our model correctly caught a DDoS attack. Nice job, model!

# So, what does this tell us? Our model is pretty cautious - it never misses a real attack, which is super important for security. But it does tend to cry wolf a bit, flagging some normal traffic as attacks.
# In real-world terms, it means our security team might have to check out some false alarms, but they can rest easy knowing that no actual attacks are slipping through unnoticed.
# Overall, it's doing a solid job, but there's room for improvement in reducing those false alarms without compromising its perfect record of catching real attacks.


# What we're looking at:
# This heatmap shows how different features in our network traffic data relate to each other. Each square represents the correlation between two features.
# The colors:
# Red squares mean the features are positively correlated – they tend to increase or decrease together.
# Blue squares indicate negative correlation – when one goes up, the other tends to go down.
# White or light-colored squares suggest little to no correlation.

# The diagonal line:
# The bright red diagonal line from top-left to bottom-right is where each feature correlates with itself (which is always 1, hence the bright red).
# Key observations:

# There are several clusters of red squares, indicating groups of features that are closely related.
# For example, "Total Fwd Packets" is strongly correlated with "Fwd Packet Length Total" and "Flow Bytes/s".
# Some features like "Protocol" and "Fwd Header Length" don't seem to correlate strongly with many other features (mostly light colors in their rows/columns).


# What this means for DDoS detection:

# Strongly correlated features might be providing redundant information. We might be able to simplify our model by using only one from each highly correlated group.
# Features with unique patterns (less correlation with others) might be particularly useful for distinguishing normal traffic from attacks.


# Interesting patterns:

# The bottom-right corner shows a cluster of highly correlated features related to "Idle" time statistics.
# "SYN Flag Count" doesn't correlate strongly with many other features, which could make it a unique indicator for certain types of attacks.


# What to watch out for:

# Very high correlations between features that should be independent might indicate data leakage or preprocessing issues.
# This heatmap helps us understand the relationships in our data, which is crucial for building an effective DDoS detection model and for interpreting its decisions.


# What it is:
# This graph shows how well our DDoS detection model can distinguish between normal traffic and attacks at different threshold settings.
# The axes:

# X-axis (False Positive Rate): This shows the proportion of normal traffic incorrectly flagged as attacks.
# Y-axis (True Positive Rate): This shows the proportion of actual attacks correctly identified.


# The blue dashed line:
# This represents a random guess model (50-50 chance). Any useful model should be above this line.
# The orange curve:
# This is our model's performance. The higher it goes towards the top-left corner, the better.
# AUC (Area Under the Curve):
# The score of 0.92 means our model is doing quite well! It's correctly identifying 92% of cases.
# What the shape tells us:

# The sharp rise at the left: Our model is great at catching many true attacks with very few false alarms.
# The flat top: Once we catch most attacks, increasing sensitivity mostly adds false alarms.


# In practical terms:

# We can catch about 70% of attacks with almost no false alarms.
# To catch more attacks, we'd need to accept more false positives.
# There's a sweet spot where we maximize caught attacks while minimizing false alarms.


# Overall performance:
# This ROC curve shows our model is doing a very good job. It's much better than random guessing and can be tuned to balance between catching attacks and minimizing false alarms effectively.

# In summary, this graph shows our DDoS detection model is performing well, with the flexibility to adjust its sensitivity based on whether we prioritize catching all attacks or minimizing false alarms.

# This image shows five different distribution plots, each representing a different aspect of network traffic data. The plots are labeled 0 and 1, which likely represent two different classes or types of network flows. Let's break down each plot:

# Distribution of Protocol:

# Shows two main spikes at values around 6-7 and 17.
# Class 1 (orange) is more prevalent in both spikes.
# This likely represents different network protocols used (e.g., TCP, UDP, HTTP).


# Distribution of Flow Duration:

# X-axis shows duration up to 1e8 (100 million) units (possibly microseconds or milliseconds).
# Class 0 (blue) has a large spike near 0, suggesting many short-duration flows.
# Class 1 (orange) has a more spread out distribution, peaking around 0.4-0.6 of the scale.


# Distribution of Total Fwd Packets:

# Shows the number of packets sent in the forward direction.
# Most flows have a small number of packets (less than 1000).
# There's a small spike for class 0 at the higher end (around 4000 packets).


# Distribution of Total Backward Packets:

# Similar to forward packets, but for the reverse direction.
# The distribution is heavily skewed towards 0, indicating many flows have few or no backward packets.
# The scale extends to 8000 packets, but most activity is concentrated near 0.


# Distribution of Fwd Packets Length Total:

# Represents the total length of forward packets in a flow.
# The distribution is highly skewed, with most flows having small total lengths.
# The scale extends to 200,000 units (possibly bytes), but most activity is near 0.



# General observations:

# There are clear differences between class 0 and 1 in most plots, suggesting these classes represent distinct types of network traffic.
# Many distributions are heavily skewed, indicating that a large number of flows have small values for these metrics, with fewer flows having larger values.
# The protocol distribution suggests that certain protocols are more associated with one class than the other.

# This data could be useful for network traffic analysis, potentially for purposes such as anomaly detection, traffic classification, or understanding typical usage patterns in a network.

# This graph shows the training loss of a machine learning model over time, specifically across different epochs of training.
# Key observations:

# Y-axis: The y-axis represents the loss value, which is a measure of how well the model is performing. Lower loss values indicate better performance.
# X-axis: The x-axis shows the number of epochs, which are complete passes through the training dataset. The graph covers 9 epochs, from 0 to 8.
# Trend: The loss is consistently decreasing over time, which is a positive sign. This indicates that the model is learning and improving its performance with each epoch.
# Shape: The curve is not linear but slightly convex, showing that the rate of improvement is gradually slowing down. This is typical in machine learning training, where initial improvements are often more dramatic.
# Starting point: The loss starts at about 0.174 at epoch 0.
# Ending point: By epoch 8, the loss has decreased to approximately 0.158.
# Smoothness: The curve appears smooth without significant fluctuations, suggesting stable learning without major irregularities.

# This graph suggests that the training process is effective, as the model's performance is consistently improving. However, the slowing rate of improvement towards the later epochs might indicate that the model is approaching its optimal performance for the given architecture and dataset. If further improvement is needed, techniques like adjusting the learning rate, modifying the model architecture, or introducing more data might be considered.