import numpy as np
import matplotlib.pyplot as plt
import torch
from model import LSTMModel
from scipy.signal import butter, filtfilt

# Load data function updated to include odometry
def load_interpolated_data_with_odom(file_path='interpolated_data.npz'):
    """
    Load the interpolated IMU data and odometry data from a .npz file.

    Args:
        file_path (str): Path to the file containing interpolated data.

    Returns:
        tuple: IMU data (numpy array) and odometry data (numpy array).
    """
    try:
        print(f"Loading interpolated data and odometry from {file_path}...")
        data = np.load(file_path, allow_pickle=True)
        imu_data = data['data']
        ground_truth = data['labels']
        odometry_data = ground_truth
        print("Odometry data extracted from 'labels'.")
        print(f"Loaded {imu_data.shape[0]} IMU entries and odometry data.")
        return imu_data, odometry_data
    except FileNotFoundError:
        print(f"File {file_path} not found!")
        return None, None, None

# Load normalization parameters
def load_normalization_params(file_path='normalization_params.npz'):
    """
    Load normalization parameters for IMU and position data.

    Args:
        file_path (str): Path to the file containing normalization parameters.

    Returns:
        tuple: Mean and standard deviation for IMU and position data.
    """
    try:
        params = np.load(file_path)
        imu_mean = params['imu_mean'].mean(axis=0)
        imu_std = params['imu_std'].mean(axis=0)
        pos_mean = params['position_mean']
        pos_std = params['position_std']
        print("Normalization parameters loaded.")
        return imu_mean, imu_std, pos_mean, pos_std
    except FileNotFoundError:
        print(f"Normalization parameters file {file_path} not found!")
        return None, None, None, None

# High-pass filter function
def high_pass_filter(data, cutoff=0.15, fs=100):
    """
    Apply a high-pass filter to remove low-frequency components.

    Args:
        data (numpy array): Input data to be filtered.
        cutoff (float): Cutoff frequency for the high-pass filter.
        fs (int): Sampling frequency of the data.

    Returns:
        numpy array: Filtered data.
    """
    b, a = butter(1, cutoff / (fs / 2), btype='high')
    return filtfilt(b, a, data, axis=0)

# Smooth data function
def smooth_data(data, window_size=5):
    """
    Apply a moving average to smooth the data.

    Args:
        data (numpy array): Input data to be smoothed.
        window_size (int): Size of the moving average window.

    Returns:
        numpy array: Smoothed data.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def classical_propagation_with_odom(imu_data, odometry_data, dt=0.02):
    """
    Perform classical propagation using IMU data and odometry for correction.

    Args:
        imu_data (numpy array): IMU data containing acceleration and gyroscope readings.
        odometry_data (numpy array): Ground truth data for periodic correction.
        dt (float): Time interval between samples.

    Returns:
        numpy array: Propagated position data.
    """
    print("Performing classical propagation with odometry...")

    accel = imu_data[:, :3]
    gyro = imu_data[:, 3:6]

    accel_bias = np.mean(accel[:300], axis=0)
    accel -= accel_bias
    accel = high_pass_filter(accel, cutoff=0.15, fs=50)
    accel = np.array([smooth_data(accel[:, i]) for i in range(accel.shape[1])]).T
    accel = np.clip(accel, -4, 4)

    num_samples = accel.shape[0]
    position = np.zeros((num_samples, 3))
    velocity = np.zeros((num_samples, 3))

    # Set initial position
    position[0] = [-2, 0, 0]

    for i in range(1, num_samples):
        velocity[i] = velocity[i - 1] + 0.5 * (accel[i - 1] + accel[i]) * dt

        position[i] = position[i - 1] + 0.5 * (velocity[i - 1] + velocity[i]) * dt

        # Periodic correction using odometry
        if i % 50 == 0: 
            position[i] = odometry_data[i, :3] 
            velocity[i] = odometry_data[i, 3:6]

    print(f"Final position with odometry correction: {position[-1]}")
    return position

def normalize_imu_data(imu_data, imu_mean, imu_std):
    """
    Normalize IMU data using the mean and standard deviation.

    Args:
        imu_data (numpy array): IMU data to be normalized.
        imu_mean (numpy array): Mean values for normalization.
        imu_std (numpy array): Standard deviation values for normalization.

    Returns:
        numpy array: Normalized IMU data.
    """
    return (imu_data - imu_mean) / imu_std

def denormalize_positions(positions, mean, std):
    """
    Denormalize positions using the provided mean and standard deviation.

    Args:
        positions (numpy array): Normalized positions to be denormalized.
        mean (numpy array): Mean values used during normalization.
        std (numpy array): Standard deviation values used during normalization.

    Returns:
        numpy array: Denormalized positions.
    """
    return (positions * std) + mean

def lstm_prediction(imu_data, model_path='model_seq50_100epochs.pth', sequence_length=50, imu_mean=None, imu_std=None, pos_mean=None, pos_std=None):
    """
    Predict positions using a pretrained LSTM model.

    Args:
        imu_data (numpy array): Input IMU data.
        model_path (str): Path to the pretrained LSTM model.
        sequence_length (int): Sequence length for the LSTM input.
        imu_mean (numpy array): Mean of IMU data for normalization.
        imu_std (numpy array): Standard deviation of IMU data for normalization.
        pos_mean (numpy array): Mean of position data for denormalization.
        pos_std (numpy array): Standard deviation of position data for denormalization.

    Returns:
        numpy array: Predicted positions.
    """
    imu_data = normalize_imu_data(imu_data, imu_mean, imu_std)

    print("Loading pretrained LSTM model...")
    model = LSTMModel(input_size=10, hidden_size=128, output_size=7, num_layers=2, dropout_rate=0.2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("Predicting using LSTM model...")
    predictions = []
    for i in range(sequence_length, len(imu_data)):
        imu_sequence = imu_data[i - sequence_length:i]
        imu_tensor = torch.tensor(imu_sequence, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(imu_tensor).squeeze(0).numpy()
        predictions.append(pred[:3])
    predictions = np.array(predictions)

    predictions = denormalize_positions(predictions, pos_mean, pos_std)
    print("LSTM prediction completed.")
    return predictions

def main():
    """
    Main function to run classical propagation and LSTM predictions.
    """
    imu_data, odometry_data = load_interpolated_data_with_odom()
    if imu_data is None or odometry_data is None:
        print("Failed to load data. Exiting...")
        return
    
    imu_mean, imu_std, pos_mean, pos_std = load_normalization_params()
    if imu_mean is None or imu_std is None:
        print("Failed to load normalization parameters. Exiting...")
        return
    
    classical_pos = classical_propagation_with_odom(imu_data, odometry_data)

    lstm_pos = lstm_prediction(imu_data,model_path='model_seq50_100epochs.pth', sequence_length=50,
                               imu_mean=imu_mean, imu_std=imu_std, pos_mean=pos_mean, pos_std=pos_std)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(classical_pos[:, 0], classical_pos[:, 1], 'r--', label='Classical Propagation with Odometry correction') 
    axes[0].set_title("Classical Propagation")
    axes[0].set_xlabel("X Position")
    axes[0].set_ylabel("Y Position")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(lstm_pos[:, 0], lstm_pos[:, 1], 'b-', label='LSTM Prediction')
    axes[1].set_title("LSTM Prediction")
    axes[1].set_xlabel("X Position")
    axes[1].set_ylabel("Y Position")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()