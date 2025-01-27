import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from bagpy import bagreader
from tqdm import tqdm
from datetime import datetime

class LSTMModel(nn.Module):
    """
    LSTM-based neural network for sequential data prediction.

    Args:
        input_size (int): Number of features in the input data.
        hidden_size (int): Number of hidden units in the LSTM layer.
        output_size (int): Number of features in the output data.
        num_layers (int, optional): Number of stacked LSTM layers. Defaults to 2.
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Use the output from the last LSTM cell
        return out

class RosbagDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and preprocessing rosbag data.

    Args:
        rosbag_file (str): Path to the rosbag file.
        imu_topic (str): Topic name for IMU data in the rosbag.
        odom_topic (str): Topic name for odometry data in the rosbag.
        sequence_length (int, optional): Length of input sequences. Defaults to 50.
        save_file (str, optional): File path for saving synchronized data. Defaults to "interpolated_data.npz".
    """
    def __init__(self, rosbag_file, imu_topic, odom_topic, sequence_length=50, save_file="interpolated_data.npz"):
        self.data = []
        self.labels = []

        if os.path.exists(save_file):
            print(f"Loading synchronized data from {save_file}...")
            loaded_data = np.load(save_file, allow_pickle=True)
            self.data = loaded_data['data']
            self.labels = loaded_data['labels']
            print(f"Loaded {len(self.data)} synchronized entries.")
        else:
            bag = bagreader(rosbag_file)
            imu_csv = bag.message_by_topic(imu_topic)
            odom_csv = bag.message_by_topic(odom_topic)

            imu_data = pd.read_csv(imu_csv)
            odom_data = pd.read_csv(odom_csv)

            print(f"Original IMU entries: {len(imu_data)}, Original Odom entries: {len(odom_data)}")

            # Synchronize timestamps and interpolate odometry data
            imu_data['timestamp'] = imu_data['header.stamp.secs'] + imu_data['header.stamp.nsecs'] * 1e-9
            odom_data['timestamp'] = odom_data['header.stamp.secs'] + odom_data['header.stamp.nsecs'] * 1e-9

            interpolated_odom = []
            for imu_time in tqdm(imu_data['timestamp'], desc="Synchronizing Data", unit="timestamp"):
                if imu_time < odom_data['timestamp'].iloc[0]:
                    interpolated_values = odom_data.iloc[0][['pose.pose.position.x', 'pose.pose.position.y', 'pose.pose.position.z',
                                                            'pose.pose.orientation.x', 'pose.pose.orientation.y', 'pose.pose.orientation.z', 'pose.pose.orientation.w']].values
                elif imu_time > odom_data['timestamp'].iloc[-1]:
                    interpolated_values = odom_data.iloc[-1][['pose.pose.position.x', 'pose.pose.position.y', 'pose.pose.position.z',
                                                            'pose.pose.orientation.x', 'pose.pose.orientation.y', 'pose.pose.orientation.z', 'pose.pose.orientation.w']].values
                else:
                    before = odom_data[odom_data['timestamp'] <= imu_time].iloc[-1]
                    after = odom_data[odom_data['timestamp'] >= imu_time].iloc[0]
                    if before['timestamp'] == after['timestamp']:
                        interpolated_values = before[['pose.pose.position.x', 'pose.pose.position.y', 'pose.pose.position.z',
                                                    'pose.pose.orientation.x', 'pose.pose.orientation.y', 'pose.pose.orientation.z', 'pose.pose.orientation.w']].values
                    else:
                        ratio = (imu_time - before['timestamp']) / (after['timestamp'] - before['timestamp'])
                        interpolated_values = before[['pose.pose.position.x', 'pose.pose.position.y', 'pose.pose.position.z',
                                                    'pose.pose.orientation.x', 'pose.pose.orientation.y', 'pose.pose.orientation.z', 'pose.pose.orientation.w']].values + \
                                    ratio * (after[['pose.pose.position.x', 'pose.pose.position.y', 'pose.pose.position.z',
                                                    'pose.pose.orientation.x', 'pose.pose.orientation.y', 'pose.pose.orientation.z', 'pose.pose.orientation.w']].values - 
                                            before[['pose.pose.position.x', 'pose.pose.position.y', 'pose.pose.position.z',
                                                    'pose.pose.orientation.x', 'pose.pose.orientation.y', 'pose.pose.orientation.z', 'pose.pose.orientation.w']].values)
                interpolated_odom.append(interpolated_values)

            self.data = imu_data[['linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z',
                                'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z',
                                'orientation.x', 'orientation.y', 'orientation.z', 'orientation.w']].values
            self.labels = np.array(interpolated_odom)

            print(f"Saving synchronized data to {save_file}...")
            np.savez(save_file, data=self.data, labels=self.labels)
            print(f"Synchronized entries: {len(self.data)}")
    
        self.sequence_length = sequence_length
        self.data = self._create_sequences(self.data, sequence_length)
        self.labels = self.labels[sequence_length - 1:]
    
    def _create_sequences(self, data, sequence_length):
        """
        Create sequences from the input data.

        Args:
            data (np.ndarray): Input data array.
            sequence_length (int): Length of each sequence.

        Returns:
            np.ndarray: Array of sequences.
        """
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

def create_excel_log():
    """
    Creates an Excel file to log training hyperparameters and metrics.

    The file contains two sheets:
    1. Hyperparameters: Documents the model's configuration.
    2. Training_Log: Stores epoch-wise loss values for training and testing.

    Returns:
        str: The name of the created Excel file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file = f'training_log_{timestamp}.xlsx'

    def format_value(val):
        return f"{val:.5f}" if isinstance(val, float) else str(val)
    
    # Create hyperparameters DataFrame
    hyperparams = {
        'Parameter': ['input_size', 'hidden_size', 'output_size', 'num_layers', 
                     'batch_size', 'num_epochs', 'learning_rate', 'weight_decay', 
                     'sequence_length'],
        'Value': [format_value(input_size), format_value(hidden_size), format_value(output_size),
                  format_value(num_layers), format_value(batch_size), format_value(num_epochs), 
                  format_value(learning_rate), format_value(weight_decay), format_value(sequence_length)]
    }
    
    # Create Excel writer
    with pd.ExcelWriter(excel_file) as writer:
        # Write hyperparameters
        pd.DataFrame(hyperparams).to_excel(writer, sheet_name='Hyperparameters', index=False)
        # Create empty training log sheet
        pd.DataFrame(columns=['Epoch', 'Train Loss', 'Test Loss']).to_excel(
            writer, sheet_name='Training_Log', index=False)
    
    return excel_file

if __name__ == '__main__':
    """
    Main script for training an LSTM model on synchronized IMU and odometry data.

    Steps:
    1. Define model hyperparameters and file paths.
    2. Preprocess and normalize input data.
    3. Split data into training and testing sets.
    4. Train the LSTM model using the training set.
    5. Validate the model using the test set.
    6. Log the training process and save the trained model.

    Notes:
    - Ensure the rosbag file path and topic names are correctly set.
    - Normalization parameters are saved for ulterior analysis.
    """
    # Parameters
    rosbag_file = "/Users/matteocoletta/Desktop/UNIVERSITA/UNIUD/AI & CyberSecurity/CORSI/II Anno/Robot Navigation/Final Projet/turtlebot3_imu_odom5_2025-01-17-01-32-02.bag"
    imu_topic = "/imu"
    odom_topic = "/odom"
    input_size = 10  # linear_acceleration.x, linear_acceleration.y, linear_acceleration.z, angular_velocity.x, angular_velocity.y, angular_velocity.z, orientation.x, orientation.y, orientation.z, orientation.w
    hidden_size = 128
    output_size = 7  # pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w
    num_layers = 2
    batch_size = 16
    num_epochs = 100
    learning_rate = 0.0001
    weight_decay = 1e-5
    sequence_length = 50

    # Prepare the dataset and dataloader
    dataset = RosbagDataset(rosbag_file, imu_topic, odom_topic, sequence_length)

    # Split the dataset into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        dataset.data, dataset.labels, test_size=0.2, random_state=42
    )

    train_data = np.array(train_data, dtype=np.float32)
    test_data = np.array(test_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.float32)

    # Normalize data
    mean, std = train_data.mean(axis=0), train_data.std(axis=0)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    # Normalize position labels
    position_mean, position_std = train_labels[:, :3].mean(axis=0), train_labels[:, :3].std(axis=0)
    train_labels[:, :3] = (train_labels[:, :3] - position_mean) / position_std
    test_labels[:, :3] = (test_labels[:, :3] - position_mean) / position_std

    # Save normalization parameters
    np.savez("normalization_params.npz", 
         imu_mean=mean, imu_std=std, 
         position_mean=position_mean, position_std=position_std)
    print("Normalization parameters saved to 'normalization_params.npz'")


    # Create DataLoaders for train and test sets
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_data), torch.tensor(train_labels)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(test_data), torch.tensor(test_labels)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # Training function with validation
    def train_eval_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, save_path="model.pth"):
        """
        Trains and evaluates the LSTM model with early stopping.

        Args:
            model (nn.Module): The LSTM model.
            train_loader (DataLoader): DataLoader for training data.
            test_loader (DataLoader): DataLoader for test data.
            criterion (nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for model training.
            num_epochs (int, optional): Maximum number of training epochs. Defaults to 10.
            save_path (str, optional): Path to save the trained model. Defaults to "model.pth".
        """
        excel_file = create_excel_log()
        training_log = []

        best_loss = float('inf')
        patience = 5  # Number of epochs to wait before stopping
        counter = 0
        
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {100 * total_train_loss / len(train_loader):.2f}%")

            # Validation
            model.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_test_loss += loss.item()
            val_loss = total_test_loss / len(test_loader)

            print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {100 * total_test_loss / len(test_loader):.2f}%")

             # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), 'best_model.pth')  # Save best model
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
            # Log metrics
            log_entry = {
                'Epoch': epoch + 1,
                'Train Loss': f"{total_train_loss / len(train_loader):.5f}",
                'Test Loss': f"{total_test_loss / len(test_loader):.5f}"
            }
            training_log.append(log_entry)
            
            # Update Excel file
            with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                pd.DataFrame(training_log).to_excel(
                    writer, 
                    sheet_name='Training_Log',
                    startrow=1,  # Start after header
                    header=False,  # Don't write headers again
                    index=False
                )

        # Save last model weights
        print(f"Saving last model to {save_path}...")
        torch.save(model.state_dict(), save_path)

    # Initialize the model, criterion, and optimizer
    model = LSTMModel(input_size, hidden_size, output_size, num_layers, dropout_rate=0.2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train the model with validation
    train_eval_model(model, train_loader, test_loader, criterion, optimizer, num_epochs)