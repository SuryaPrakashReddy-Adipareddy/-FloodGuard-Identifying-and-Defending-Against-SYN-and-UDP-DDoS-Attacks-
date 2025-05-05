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
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import optuna
import shap
import logging
from datetime import datetime
import wandb
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Dict, Optional, List
from pathlib import Path
warnings.filterwarnings('ignore')

class Config:
    """Configuration class for hyperparameters and settings."""
    def __init__(self):
        self.RANDOM_SEED = 42
        self.NUM_EPOCHS = 20
        self.BATCH_SIZE = 64
        self.LEARNING_RATE = 0.001
        self.PATIENCE = 7
        self.MIN_DELTA = 1e-4
        self.SMOTE_K_NEIGHBORS = 5
        self.FOCAL_LOSS_ALPHA = 0.25
        self.FOCAL_LOSS_GAMMA = 3
        self.VALIDATION_SPLIT = 0.2
        self.NUM_OPTUNA_TRIALS = 50
        self.GRAD_CLIP_VALUE = 1.0
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.RANDOM_SEED)
        np.random.seed(self.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.RANDOM_SEED)

class DataModule:
    """Handles all data loading and preprocessing operations."""
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)

    def load_and_preprocess_data(self, data_paths: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess the datasets."""
        try:
            # Load datasets
            dfs = {name: pd.read_csv(path) for name, path in data_paths.items()}
            
            # Clean and combine datasets
            train_df = pd.concat([
                self._clean_dataframe(dfs['syn_train']),
                self._clean_dataframe(dfs['udp_train'])
            ])
            test_df = pd.concat([
                self._clean_dataframe(dfs['syn_test']),
                self._clean_dataframe(dfs['udp_test'])
            ])
            
            # Process labels
            train_df, test_df = self._process_labels(train_df, test_df)
            
            return train_df, test_df
            
        except Exception as e:
            self.logger.error(f"Error in data loading: {str(e)}")
            raise

    def prepare_model_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple:
        """Prepare data for model training."""
        try:
            # Split features and labels
            X_train = train_df.drop('Label', axis=1).values
            y_train = train_df['Label'].values
            X_test = test_df.drop('Label', axis=1).values
            y_test = test_df['Label'].values
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Apply SMOTE
            smote = SMOTE(random_state=self.config.RANDOM_SEED, 
                         k_neighbors=self.config.SMOTE_K_NEIGHBORS)
            X_train_resampled, y_train_resampled = smote.fit_resample(
                X_train_scaled, y_train
            )
            
            # Reshape for LSTM
            X_train_lstm = self._reshape_for_lstm(X_train_resampled)
            X_test_lstm = self._reshape_for_lstm(X_test_scaled)
            
            return X_train_lstm, X_test_lstm, y_train_resampled, y_test

        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            raise

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess a dataframe."""
        df_cleaned = df.copy()
        df_cleaned = df_cleaned.dropna()
        df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan).dropna()
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_columns] = df_cleaned[numeric_columns].astype(np.float32)
        return df_cleaned

    @staticmethod
    def _process_labels(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process and verify labels in datasets."""
        label_map = {'Benign': 0, 'Syn': 1, 'UDP': 1}
        for df in [train_df, test_df]:
            df['Label'] = df['Label'].map(label_map)
            df = df.dropna(subset=['Label'])
            df['Label'] = df['Label'].astype(int)
        return train_df, test_df

    @staticmethod
    def _reshape_for_lstm(X: np.ndarray) -> np.ndarray:
        """Reshape data for LSTM input (samples, timesteps, features)."""
        return X.reshape(X.shape[0], 1, X.shape[1])

class LSTMModel(nn.Module):
    """LSTM model with attention mechanism for DDoS detection."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 num_classes: int, dropout: float = 0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
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

    def attention_net(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention mechanism to LSTM output."""
        attention_weights = self.attention(lstm_output)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        attn_output, attention_weights = self.attention_net(lstm_out)
        out = self.fc_layers(attn_output)
        return out, attention_weights

class FocalLoss(nn.Module):
    """Focal Loss implementation for dealing with class imbalance."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = 1e-6

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss."""
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
        return loss

class ModelTrainer:
    """Handles model training and evaluation."""
    def __init__(self, model: nn.Module, config: Config, output_dir: str):
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, train_loader: torch.utils.data.DataLoader, 
              val_data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[List[float], List[float]]:
        """Train the model."""
        wandb.init(project="ddos_detection", config=self.config.__dict__)
        writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard_logs'))
        
        criterion = FocalLoss(alpha=self.config.FOCAL_LOSS_ALPHA, 
                            gamma=self.config.FOCAL_LOSS_GAMMA)
        optimizer = optim.AdamW(self.model.parameters(), 
                              lr=self.config.LEARNING_RATE, 
                              weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                    patience=3, verbose=True)
        
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Training
            self.model.train()
            epoch_losses = []
            
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS}') as pbar:
                for batch_X, batch_y in pbar:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs, _ = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.GRAD_CLIP_VALUE
                    )
                    
                    optimizer.step()
                    epoch_losses.append(loss.item())
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Validation
            val_loss = self._validate(val_data, criterion)
            
            # Logging
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            self._log_metrics(epoch, avg_train_loss, val_loss, writer)
            
            # Model checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)
            
            scheduler.step(val_loss)
        
        writer.close()
        wandb.finish()
        return train_losses, val_losses

    def _validate(self, val_data: Tuple[torch.Tensor, torch.Tensor], 
                 criterion: nn.Module) -> float:
        """Validate the model."""
        self.model.eval()
        X_val, y_val = val_data
        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
        
        with torch.no_grad():
            outputs, _ = self.model(X_val)
            val_loss = criterion(outputs, y_val)
        
        return val_loss.item()

    def _log_metrics(self, epoch: int, train_loss: float, val_loss: float, 
                    writer: SummaryWriter) -> None:
        """Log metrics to wandb and tensorboard."""
        self.logger.info(
            f'Epoch [{epoch+1}/{self.config.NUM_EPOCHS}], '
            f'Train Loss: {train_loss:.4f}, '
            f'Val Loss: {val_loss:.4f}'
        )
        
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss
        })
        
        writer.add_scalars('Losses', {
            'train': train_loss,
            'val': val_loss
        }, epoch)

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.output_dir, 
            f'model_checkpoint_epoch_{epoch}_loss_{val_loss:.4f}.pt'
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)

def main():
    """Main execution function."""
    # Initialize configuration
    config = Config()
    
    # Create output directory and setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"ddos_analysis_output_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / f'ddos_detection_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles model evaluation and performance visualization."""
    def __init__(self, model: nn.Module, config: Config, output_dir: Path):
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)

    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate model performance on test set."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(self.device), y_test.to(self.device)
                outputs, _ = self.model(X_test)
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_test.cpu().numpy())

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        results = {
            'accuracy': np.mean(all_preds == all_labels),
            'precision': np.sum((all_preds == 1) & (all_labels == 1)) / (np.sum(all_preds == 1) + 1e-10),
            'recall': np.sum((all_preds == 1) & (all_labels == 1)) / (np.sum(all_labels == 1) + 1e-10),
        }
        results['f1'] = 2 * (results['precision'] * results['recall']) / (results['precision'] + results['recall'] + 1e-10)
        
        return results

    def plot_metrics(self, test_loader: torch.utils.data.DataLoader) -> None:
        """Generate and save performance visualization plots."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(self.device), y_test.to(self.device)
                outputs, _ = self.model(X_test)
                probs = torch.sigmoid(outputs)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y_test.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.savefig(self.output_dir / 'roc_curve.png')
        plt.close()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(self.output_dir / 'pr_curve.png')
        plt.close()

def main():
    """Main execution function."""
    try:
        # Initialize configuration
        config = Config()
        
        # Create output directory and setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"ddos_analysis_output_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(output_dir / f'ddos_detection_{timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        
        # Define data paths
        data_paths = {
            'syn_train': 'Syn-tsting.csv',
            'udp_train': 'data/udp_flood_train.csv',
            'syn_test': 'data/syn_flood_test.csv',
            'udp_test': 'data/udp_flood_test.csv'
        }
        
        # Initialize data module and load data
        data_module = DataModule(config)
        train_df, test_df = data_module.load_and_preprocess_data(data_paths)
        X_train_lstm, X_test_lstm, y_train_resampled, y_test = data_module.prepare_model_data(train_df, test_df)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train_lstm),
            torch.FloatTensor(y_train_resampled).reshape(-1, 1)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test_lstm),
            torch.FloatTensor(y_test).reshape(-1, 1)
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False
        )
        
        # Initialize model
        input_size = X_train_lstm.shape[2]
        model = LSTMModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            num_classes=1
        )
        
        # Train model
        trainer = ModelTrainer(model, config, output_dir)
        train_losses, val_losses = trainer.train(train_loader, (
            torch.FloatTensor(X_test_lstm),
            torch.FloatTensor(y_test).reshape(-1, 1)
        ))
        
        # Evaluate model
        evaluator = ModelEvaluator(model, config, output_dir)
        metrics = evaluator.evaluate(test_loader)
        evaluator.plot_metrics(test_loader)
        
        # Log final results
        logger.info("Training completed successfully")
        logger.info(f"Final metrics: {metrics}")
        
        # Save final model
        torch.save(model.state_dict(), output_dir / 'final_model.pt')
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()