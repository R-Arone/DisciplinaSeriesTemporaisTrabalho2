import os
import random

import torch
from torch.utils.data import Dataset
import mne
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, data_files, labels_files, n_past, segmentation_size_sec = 5, augmentations=None, random_offset=True):
        """
        :param data: Raw time series data (shape: [num_samples, num_timepoints, n_sensors])
        :param labels: Corresponding labels for classification (shape: [num_samples])
        :param n_sensors: Number of sensor observations at each time point
        :param n_past: Number of past time points to use for prediction (time lag)
        :param augmentations: A list of augmentation functions to apply to data (e.g., noise, normalization)
        :param random_offset: Whether to apply a random offset when selecting the past observations
        """
        self.data = []
        self.annotations = []
        self.index_data = []
        self.data_files = data_files
        self.labels_files = labels_files
        self.idx_offset = []
        assert len(data_files) == len(labels_files), "Number of data files and labels files must match"
        for i, obj in enumerate(zip(self.data_files, self.labels_files)):
            data_file, labels_file = obj
            assert os.path.exists(data_file), f"Data file {data_file} does not exist"
            assert os.path.exists(labels_file), f"Labels file {labels_file} does not exist"
            annotation, data = self.read_data(data_file, labels_file)
            self.data.append(data)
            self.annotations += list(annotation)
            self.index_data += [i]*len(annotation)
            self.idx_offset.append(len(annotation))

        self.n_past = n_past
        self.augmentations = augmentations
        self.random_offset = random_offset
        self.segmentation_size_sec = segmentation_size_sec
        self.segmentation_size = int(self.segmentation_size_sec*self.fs)

    def read_annotations(self, file):
        with open(file, 'r') as f:
            annotations = f.readlines()
        annotations = [int(x.strip()) for x in annotations[1:]]
        return annotations

    def read_data(self, file, labels_file):
        data = mne.io.read_raw_edf(file, preload=True)
        raw_data = data.get_data()

        # Load labels
        info = data.info
        channels = data.ch_names
        annotations = self.read_annotations(labels_file)

        reference_channels = self.group_channels(data)

        fs = int(info['sfreq']) #Hz
        #data.filter(0.5, 25, fir_design='firwin')
        raw_data = self.get_data(data, reference_channels, ['eeg', 'eog'])
        #Downsample from 200Hz to 50Hz
        #raw_data = sg.decimate(raw_data, 4, axis=1)
        new_fs = fs
        segmentation_size = int(5*new_fs) #30 seconds
        self.fs = new_fs
        print(f"Data shape: {raw_data.shape}, fs: {new_fs}, segmentation size: {segmentation_size}, annotations: {len(annotations)}")

        #scaler = StandardScaler()

        # Fit the scaler for normalization (across all features)
        data_reshaped = raw_data.T
        data_reshaped = self.z_normalize(data_reshaped)
        #data_reshaped = scaler.fit_transform(data_reshaped)
        #Apply rolling z-normalization on each channel separately and then reshape the data back
        #data_reshaped = np.array([self.rolling_z_normalize(data_reshaped[:, i]) for i in range(data_reshaped.shape[1])]).T

        #If the data is not divisible by the segmentation size, we pad it with zeros
        pad_size = segmentation_size - (data_reshaped.shape[0] % segmentation_size)
        data_reshaped = np.pad(data_reshaped, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)

        #Remove annotations that are outside the data
        if len(annotations) > data_reshaped.shape[0] // segmentation_size:
            annotations = annotations[:data_reshaped.shape[0] // segmentation_size]

        #Change annotations equals to -1 to 0
        annotations = [0 if x == -1 else x for x in annotations]

        print(f"Data shape after padding: {data_reshaped.shape}, annotations: {len(annotations)}")

        return annotations, data_reshaped

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data_idx = self.index_data[idx]
        # Label for classification
        label = self.annotations[idx]
        # Get the data index (accounting for the offset)
        idx = idx - sum(self.idx_offset[:data_idx])
        # Random offset (shift in past time)
        if self.random_offset:
            # Choose a random offset (between 0 and n_past)
            offset = random.randint(0, self.n_past)
        else:
            offset = 0

        # Get the past n_past time points
        start_idx = max(0, idx - self.n_past + offset)
        end_idx = idx + 1

        # Data slice [start_idx:end_idx] to get n_past observations (time steps)
        data_start_tstamp = int(start_idx*self.segmentation_size)
        data_end_tstamp = int(end_idx*self.segmentation_size)
        time_window_data = self.data[data_idx][data_start_tstamp:data_end_tstamp]

        # If not enough time steps are available at the beginning, pad the data
        if time_window_data.shape[0] < (self.n_past + 1)*self.segmentation_size:
            pad_size = (self.n_past + 1)*self.segmentation_size - time_window_data.shape[0]
            time_window_data = np.pad(time_window_data, ((pad_size, 0), (0, 0)), mode='constant', constant_values=0)

        # Apply augmentations
        if self.augmentations:
            for aug in self.augmentations:
                time_window_data = aug(time_window_data)
        
        #time_window_data = StandardScaler().fit_transform(time_window_data)

        # Convert to torch tensor
        return torch.tensor(time_window_data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
    def get_data(self, data, reference_channels, channel_type):
        """Get a single vector of signals by type.
    
        Args:
            channel_type (str or list of str): Type(s) of the channel(s).
    
        Raises:
            ValueError: If a channel type does not exist.
    
        Returns:
            np.ndarray: Concatenated data vector for the specified channel type(s).
        """
        if isinstance(channel_type, str):
            channel_type = [channel_type]

        if not all(ct in reference_channels.keys() for ct in channel_type):
            invalid_types = [ct for ct in channel_type if ct not in reference_channels.keys()]
            raise ValueError(f"The following channel type(s) do not exist: {', '.join(invalid_types)}")

        # Get all signals and resample to the length of the longest signal
        signals = [data.get_data(reference_channels[ct]) for ct in channel_type]
        for i, signal in enumerate(signals):
            print(f"Signal shape: {signal.shape}")
        return np.concatenate(signals, axis=0)

    def group_channels(self, data):
        reference_channels_regex = {}

        reference_channels_regex['eeg'] = r'(CZ|CZ2|CZ-A1|CZ2-A1|O[1-9]|O[1-9]-A[1-9]|FP[1-9]|FP[1-9]-A[1-9])'
        reference_channels_regex['ecg'] = r'(ECG)'
        reference_channels_regex['emg'] = r'(EMG[1-9]|EMG1|EMG2|EMG3|EMG[0-9])'
        reference_channels_regex['eog'] = r'(EOG[1-9]|EOG1|EOG2)'

        return dict(map(lambda x: (x, mne.pick_channels_regexp(data.ch_names, regexp=reference_channels_regex[x])), reference_channels_regex.keys()))

    @staticmethod
    def add_noise(data, noise_factor=0.01):
        """
        Add Gaussian noise to the data.
        """
        noise = np.random.normal(0, noise_factor, data.shape)
        return data + noise
    
    @staticmethod
    def z_normalize(data):
        """
        Normalize the data to have zero mean and unit variance.
        """
        return (data - np.mean(data, axis = 0)) / np.std(data, axis = 0)
    
def custom_collate(batch):
    # Ensure that all elements are properly batched.
    data, labels = zip(*batch)
    data = torch.stack(data)  # Stack data into a single tensor
    labels = torch.stack(labels)  # Stack labels into a single tensor
    return data, labels

