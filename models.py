import torch
import torch.nn as nn
import torch.fft

class SlidingFFT(nn.Module):
    def __init__(self, fs: int, window_duration: float, stride: int):
        """
        Sliding FFT module with vectorized operations, `fs`, and stride.
        Args:
            fs (int): Sampling frequency (samples per second).
            window_duration (float): Duration of the FFT window in seconds.
            stride (int): Step size (in samples) between consecutive windows.
        """
        super(SlidingFFT, self).__init__()
        self.fs = fs
        self.window_size = int(window_duration * fs)  # Convert duration to number of samples
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the sliding FFT for each channel in a vectorized manner.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time_steps).
        Returns:
            torch.Tensor: FFT features of shape (batch_size, channels, num_windows, fft_size).
        """
        batch_size, ctime_steps, channels = x.shape
        #x = x.permute(0, 2, 1)
        #print(x.shape)

        # Compute sliding windows with the specified stride
        x_windows = x.unfold(dimension=-1, size=self.window_size, step=self.stride)  # (batch_size, channels, num_windows, window_size)

        # Perform FFT on the sliding windows
        fft_result = torch.fft.rfft(x_windows, dim=-1)  # FFT along the last dimension
        fft_magnitude = torch.abs(fft_result)  # Use magnitude of FFT

        fft_magnitude = fft_magnitude.permute(0,2,1,3)
        fft_magnitude = fft_magnitude.reshape(fft_magnitude.shape[0], fft_magnitude.shape[1], -1)
        
        return fft_magnitude
    

class Model_CNN_LSTM(nn.Module):
    def __init__(self, n_sensors, num_classes, time_frame: 5, fs: 200):
        super(Model_CNN_LSTM, self).__init__()
        self.fs = fs
        self.time_frame = time_frame
        self.n_sensors = n_sensors
        self.num_classes = num_classes
        #Do the first conv considering the timme frame, putting a conv of kernel size 0.1s
        self.conv1 = nn.Conv1d(in_channels=n_sensors, 
                               out_channels=32, 
                               kernel_size=int(fs/10), #0.1s
                               padding='same')
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) #Downsample by 2 - fs = fs_0/2
        self.conv2 = nn.Conv1d(in_channels=32, 
                               out_channels=64, 
                               kernel_size=int(fs/10), #0.2s
                               padding='same')
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) #Downsample by 2 - fs = fs_0/4

        #Third Convolutional Layer
        self.conv3 = nn.Conv1d(in_channels=64, 
                               out_channels=128, 
                               kernel_size=int(fs/4), #1s
                               padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=int(fs/4), stride = int(fs/8)) #Downsample to 0.5s

        #LSTM Layer
        self.lstm = nn.LSTM(input_size=128, 
                            hidden_size=64, 
                            num_layers=1, 
                            batch_first=True, 
                            dropout=0.2,
                            bidirectional=True)
        
        self.cnn4 = nn.Conv1d(in_channels=128, 
                               out_channels=256, 
                               kernel_size=5, #2s
                               padding='same')
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=9, stride=9) #Downsample to one feature per segment (5s)

        #Dropouts
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.3)


        
        # Fully connected layer for classification
        self.fc = nn.Linear(256, num_classes)

        #Start the weights with xavier
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)



    def forward(self, x):
        # Shape of x: (batch_size, time_steps, n_sensors)
        # 1D Convolutional Layer (transforming the data from [batch_size, time_steps, n_sensors] to [batch_size, cnn_filters, time_steps])
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        # LSTM forward pass
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        #
        ## Pass through the CNN4
        x = lstm_out.permute(0, 2, 1)
        x = self.cnn4(x)

        x = self.bn4(x)
        x = self.pool4(x)
        x = self.dropout4(x)


        #Add ativation function
        x = torch.relu(x)
        # Get only the last value from the cnn output
        x = x[:, :, -1]
        x = self.fc(x)
        return x


class CNN_Reparametrization_Trick(nn.Module):
    def __init__(self, **kwargs):
        super(CNN_Reparametrization_Trick, self).__init__()
        self.mean_conv = nn.Conv1d(**kwargs)
        self.log_var_conv = nn.Conv1d(**kwargs)


    def forward(self, x):
        mean = self.mean_conv(x)
        log_var = self.log_var_conv(x)
         # Apply the reparameterization trick only during training
        if self.training:
            # During training, add noise scaled by the log variance
            return mean + torch.randn_like(mean) * torch.exp(0.5 * log_var)
        else:
            # During evaluation, return the mean (no noise added)
            return mean


class Model_CNN_Reparametrization_LSTM(nn.Module):
    def __init__(self, n_sensors, num_classes, time_frame: 5, fs: 200):
        super(Model_CNN_Reparametrization_LSTM, self).__init__()
        self.fs = fs
        self.time_frame = time_frame
        self.n_sensors = n_sensors
        self.num_classes = num_classes
        #Do the first conv considering the timme frame, putting a conv of kernel size 0.1s
        self.conv1 = nn.Conv1d(in_channels=n_sensors, 
                               out_channels=32, 
                               kernel_size=int(fs/10), #0.1s
                               padding='same')
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) #Downsample by 2 - fs = fs_0/2
        self.conv2 = CNN_Reparametrization_Trick(in_channels=32, 
                               out_channels=64, 
                               kernel_size=int(fs/10), #0.2s
                               padding='same')
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) #Downsample by 2 - fs = fs_0/4

        #Third Convolutional Layer
        self.conv3 = nn.Conv1d(in_channels=64, 
                               out_channels=128, 
                               kernel_size=int(fs/4), #1s
                               padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=int(fs/4), stride = int(fs/8)) #Downsample to 0.5s

        #LSTM Layer
        self.lstm = nn.LSTM(input_size=128, 
                            hidden_size=64, 
                            num_layers=1, 
                            batch_first=True, 
                            dropout=0.2,
                            bidirectional=True)
        
        self.cnn4 = nn.Conv1d(in_channels=128, 
                               out_channels=256, 
                               kernel_size=5, #2s
                               padding='same')
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=9, stride=9) #Downsample to one feature per segment (5s)

        #Dropouts
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.3)


        
        # Fully connected layer for classification
        self.fc = nn.Linear(256, num_classes)

        #Start the weights with xavier
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)



    def forward(self, x):
        # Shape of x: (batch_size, time_steps, n_sensors)
        # 1D Convolutional Layer (transforming the data from [batch_size, time_steps, n_sensors] to [batch_size, cnn_filters, time_steps])
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        # LSTM forward pass
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        #
        ## Pass through the CNN4
        x = lstm_out.permute(0, 2, 1)
        x = self.cnn4(x)

        x = self.bn4(x)
        x = self.pool4(x)
        x = self.dropout4(x)


        #Add ativation function
        x = torch.relu(x)
        # Get only the last value from the cnn output
        x = x[:, :, -1]
        x = self.fc(x)
        return x



class SelfAttention(nn.Module):
  def __init__(self, input_dim):
    super(SelfAttention, self).__init__()
    self.input_dim = input_dim
    self.query = nn.Linear(input_dim, input_dim) # [batch_size, seq_length, input_dim]
    self.key = nn.Linear(input_dim, input_dim) # [batch_size, seq_length, input_dim]
    self.value = nn.Linear(input_dim, input_dim)
    self.softmax = nn.Softmax(dim=2)
   
  def forward(self, x): # x.shape (batch_size, seq_length, input_dim)
    queries = self.query(x)
    keys = self.key(x)
    values = self.value(x)

    scores = torch.bmm(queries, keys.transpose(1, 2))/(self.input_dim**0.5)
    attention = self.softmax(scores)
    weighted = torch.bmm(attention, values)
    return weighted
  
class CrossAttention(SelfAttention):

  def forward(self, x, y): # x.shape (batch_size, seq_length, input_dim)
    queries = self.query(y)
    keys = self.key(x)
    values = self.value(x)

    scores = torch.bmm(queries, keys.transpose(1, 2))/(self.input_dim**0.5)
    attention = self.softmax(scores)
    weighted = torch.bmm(attention, values)
    return weighted
  

class MultiResolutionLSTMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(MultiResolutionLSTMBlock, self).__init__()

        #Add a conv layer to reduce the the size of the input and the number of channels
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels = out_channels,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 padding='same')
        
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Downsample by 2 - fs = fs_0/2
        self.bn1 = nn.BatchNorm1d(out_channels)	
        self.dropout1 = nn.Dropout(0.1)

        self.lstm = nn.LSTM(input_size=out_channels,
                            hidden_size=out_channels,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.2,
                            bidirectional=True)
        
        #Conv to reduce the number of channels
        self.conv2 = nn.Conv1d(in_channels=out_channels*2,
                                 out_channels = out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding='same')

    def forward(self, x):
        # Shape of x: (batch_size, seq_len, n_sensors)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, seq_len, channels)
        x_out, _ = self.lstm(x)
        x_out = x_out.permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        x_out = self.conv2(x_out)
        x = x + x_out
        x = torch.relu(x)
        return x

class MultiResolutionModel(nn.Module):
    def __init__(self, n_sensors, num_classes, fs: 200, device='cuda', **kwargs):
        super(MultiResolutionModel, self).__init__()
        self.fs = fs
        self.n_sensors = n_sensors
        self.num_classes = num_classes
        self.device = device

        self.conv0 = nn.Conv1d(in_channels=n_sensors,
                                 out_channels=32,
                                 kernel_size=int(fs/20), # 0.05s
                                 padding='same')
        self.bn0 = nn.BatchNorm1d(32)
        self.pool0 = nn.MaxPool1d(kernel_size=2, stride=2)  # Downsample by 2 - fs = fs_0/2

        self.lstmresolutionblock1 = MultiResolutionLSTMBlock(in_channels=32, out_channels=64, kernel_size=int(fs/10))
        self.lstmresolutionblock2 = MultiResolutionLSTMBlock(in_channels=64, out_channels=128, kernel_size=int(fs/5))
        self.lstmresolutionblock3 = MultiResolutionLSTMBlock(in_channels=128, out_channels=256, kernel_size=int(fs/2))
        self.lstmresolutionblock4 = MultiResolutionLSTMBlock(in_channels=256, out_channels=512, kernel_size=int(fs))

        #Create decoder with 4 cnn 1d with kernel size 1 and them upsample
        self.dec_conv4 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.dec_downsample4 = nn.Identity()
        self.dec_conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1)
        self.dec_downsample3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1)
        self.dec_downsample2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dec_conv1 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=1)
        self.dec_downsample1 = nn.MaxPool1d(kernel_size=8, stride=8)

        #Conv1d to reduce the number of channels and concatenate with the output of the decoders
        self.conv1 = nn.Conv1d(in_channels=256*4, out_channels=128, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(128)
        #self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)  # Downsample by 2 - fs = fs_0/2

        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=64,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.2,
                            bidirectional=True)
        

        # Fully connected layer for classification
        self.fc = nn.Linear(128, num_classes)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                    

    def forward(self, x):
        # Shape of x: (batch_size, seq_len, n_sensors)
        batch_size, seq_len, _ = x.size()  # seq_len is inferred from input
        
        # Pass through the first convolution layer
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, n_sensors, seq_len)
        x = self.conv0(x)
        x = self.bn0(x)
        x = torch.relu(x)
        x = self.pool0(x)
        

        # Pass through the first segformer
        x =  self.lstmresolutionblock1(x)
        x2 = self.lstmresolutionblock2(x)
        x3 = self.lstmresolutionblock3(x2)
        x4 = self.lstmresolutionblock4(x3)


        #Decoder
        x = self.dec_conv1(x)
        x = self.dec_downsample1(x)
        x = torch.relu(x)
        x2 = self.dec_conv2(x2)
        x2 = self.dec_downsample2(x2)
        x2 = torch.relu(x2)
        x3 = self.dec_conv3(x3)
        x3 = self.dec_downsample3(x3)
        x3 = torch.relu(x3)
        x4 = self.dec_conv4(x4)
        x4 = self.dec_downsample4(x4)
        x4 = torch.relu(x4)


        #Concatenate the outputs
        x = torch.cat([x, x2, x3, x4], dim=1)

        #Conv1d to reduce the number of channels
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        #Pass to the lstm
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        
        # Get only the last value from the lstm output
        x = x[:, -1, :]

        x = self.fc(x)
        
        return x
    

class CrossAttentionResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super(CrossAttentionResidualBlock, self).__init__()
        self.cross_attention = CrossAttentionTimeSeriesModel(input_dim)

    def forward(self, x, y):
        x = x.permute(0,2,1)
        y = y.permute(0,2,1)
        x_att = self.cross_attention(x, y)
        x = x + x_att
        x = x.permute(0,2,1)
        x = torch.relu(x)
        return x

class TimeSeriesEmbedding(nn.Module):
    def __init__(self, input_size, embed_size):
        super(TimeSeriesEmbedding, self).__init__()
        self.fc = nn.Linear(input_size, embed_size)

    def forward(self, x):
        return self.fc(x)
    
    def initialize_weights(self):
        # Initialize weights using Xavier (Glorot) initialization for the fully connected layer
        torch.nn.init.xavier_uniform_(self.fc.weight)
        # Initialize bias to zero
        if self.fc.bias is not None:
            torch.nn.init.zeros_(self.fc.bias)
    
class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_size, num_heads)

    def forward(self, x, y):
        # sequence_1: Queries
        # sequence_2: Keys and Values
        # sequence_1 shape: (seq_len_1, batch_size, embed_size)
        # sequence_2 shape: (seq_len_2, batch_size, embed_size)
        attn_output, attn_weights = self.attn(y,  x, x)
        return attn_output, attn_weights
    
    def initialize_weights(self):
        # MultiheadAttention internally contains linear layers for Q, K, V projections.
        # Initialize their weights using Xavier
        for param in self.attn.parameters():
            if param.dim() > 1:  # If it's a weight matrix (not a bias)
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)  # Initialize biases to zero

class CrossAttentionTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, num_heads = 1):
        super(CrossAttentionTimeSeriesModel, self).__init__()
        # Embedding layers for both sequences
        self.embed_1 = TimeSeriesEmbedding(input_dim, input_dim)
        self.embed_2 = TimeSeriesEmbedding(input_dim, input_dim)
        # Cross-attention layer
        self.cross_attention = CrossAttentionLayer(input_dim, num_heads)
        self.initialize_weights()


    def forward(self, x ,y):
        # Embed both time series
        seq_1_embed = self.embed_1(x)
        seq_2_embed = self.embed_2(y)
        # Apply cross-attention
        output, attn_weights = self.cross_attention(seq_1_embed, seq_2_embed)
        return output
    
    def initialize_weights(self):
        # Initialize weights for the entire model
        # Embedding layers
        self.embed_1.initialize_weights()
        self.embed_2.initialize_weights()
        # Cross-attention layer
        self.cross_attention.initialize_weights()

class FFTLSTMBlock(nn.Module):
    def __init__(self, num_channels, fs: 200, step=50, window_size = 2, out_channels=128, hidden_size=64):
        super(FFTLSTMBlock, self).__init__()
        self.num_channels = num_channels
        self.fs = fs
        self.step = step
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.fft = SlidingFFT(fs=fs, stride=step, window_duration=window_size)
        self.in_channels = int(num_channels * (fs/2 * window_size + 1))
        self.conv1 = nn.Conv1d(in_channels=self.in_channels,
                               out_channels=self.hidden_size,
                               kernel_size=1,
                               padding='same')
        #BatchNorm
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        #Second cnn layer
        self.conv2 = nn.Conv1d(in_channels=self.hidden_size,
                               out_channels=2*self.hidden_size,
                               kernel_size=3,
                               padding='same')
        #BatchNorm
        self.bn2 = nn.BatchNorm1d(2*self.hidden_size)
        #LSTM
        self.lstm = nn.LSTM(input_size=2*self.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.2,
                            bidirectional=True)
        
        # Third cnn
        self.conv3 = nn.Conv1d(in_channels=2*self.hidden_size,
                               out_channels=self.hidden_size,
                               kernel_size=1,
                               padding='same')
        #BatchNorm
        self.bn3 = nn.BatchNorm1d(self.hidden_size)

        # 1d cnn layer for classification to be the output
        self.conv4 = nn.Conv1d(in_channels=self.hidden_size,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 padding='same')
        
        #Dropout
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)

        #initialize weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fft(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        return x


class MultiResolutionLSTMFFT(nn.Module):
    def __init__(self, num_channels, num_classes, fs: 200, device='cuda', **kwargs):
        super(MultiResolutionLSTMFFT, self).__init__()
        self.fs = fs
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.device = device

        #Calculare step and window size for the fft
        window_size = 2
        #We want to have 
        step = int(fs/10)

        self.fft_lstm_block = FFTLSTMBlock(num_channels=num_channels, fs=fs, step=step, window_size=window_size, out_channels=256, hidden_size=512)

        self.conv0 = nn.Conv1d(in_channels=num_channels,
                                 out_channels=32,
                                 kernel_size=int(fs/20), # 0.05s
                                 padding='same')
        self.bn0 = nn.BatchNorm1d(32)
        self.pool0 = nn.MaxPool1d(kernel_size=2, stride=2)  # Downsample by 2 - fs = fs_0/2

        self.lstmresolutionblock1 = MultiResolutionLSTMBlock(in_channels=32, out_channels=64, kernel_size=int(fs/20))
        self.lstmresolutionblock2 = MultiResolutionLSTMBlock(in_channels=64, out_channels=128, kernel_size=int(fs/20))
        self.lstmresolutionblock3 = MultiResolutionLSTMBlock(in_channels=128, out_channels=256, kernel_size=int(fs/20))
        self.lstmresolutionblock4 = MultiResolutionLSTMBlock(in_channels=256, out_channels=512, kernel_size=int(fs/20))

        #Create decoder with 4 cnn 1d with kernel size 1 and them upsample
        self.dec_conv4 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.dec_downsample4 = nn.Identity()
        self.dec_conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1)
        self.dec_downsample3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1)
        self.dec_downsample2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dec_conv1 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=1)
        self.dec_downsample1 = nn.MaxPool1d(kernel_size=8, stride=8)

        #Conv1d to reduce the number of channels and concatenate with the output of the decoders
        self.conv1 = nn.Conv1d(in_channels=256*4, out_channels=256, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(in_channels=256*5, out_channels=256, kernel_size=1)
        self.bn2= nn.BatchNorm1d(256)
        #self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)  # Downsample by 2 - fs = fs_0/2

        #Cross attention layers to mmerge with the fft info
        self.cross_attention1 = CrossAttentionResidualBlock(256)
        self.cross_attention2 = CrossAttentionResidualBlock(256)
        self.cross_attention3 = CrossAttentionResidualBlock(256)
        self.cross_attention4 = CrossAttentionResidualBlock(256)
        self.cross_attention5 = CrossAttentionResidualBlock(256)

        self.lstm = nn.LSTM(input_size=256,
                            hidden_size=64,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.2,
                            bidirectional=True)
        

        # Fully connected layer for classification
        self.fc = nn.Linear(128, num_classes)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                    

    def forward(self, x):
        # Shape of x: (batch_size, seq_len, n_sensors)
        batch_size, seq_len, _ = x.size()  # seq_len is inferred from input
        
        # Pass through the first convolution layer
        x_fft = self.fft_lstm_block(x)
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, n_sensors, seq_len)
        x = self.conv0(x)
        x = self.bn0(x)
        x = torch.relu(x)
        x = self.pool0(x)
        

        # Pass through the first segformer
        x =  self.lstmresolutionblock1(x)
        x2 = self.lstmresolutionblock2(x)
        x3 = self.lstmresolutionblock3(x2)
        x4 = self.lstmresolutionblock4(x3)


        #Decoder
        x = self.dec_conv1(x)
        x = self.dec_downsample1(x)
        x = torch.relu(x)
        x2 = self.dec_conv2(x2)
        x2 = self.dec_downsample2(x2)
        x2 = torch.relu(x2)
        x3 = self.dec_conv3(x3)
        x3 = self.dec_downsample3(x3)
        x3 = torch.relu(x3)
        x4 = self.dec_conv4(x4)
        x4 = self.dec_downsample4(x4)
        x4 = torch.relu(x4)


        x_cat = torch.cat([x, x2, x3, x4], dim=1)
        x_cat = self.conv1(x_cat)
        x_cat = self.bn1(x_cat)
        x_cat = torch.relu(x_cat)

        #use the cross attention to merge the outputs
        x = self.cross_attention1(x, x_fft)
        x2 = self.cross_attention2(x2, x_fft)
        x3 = self.cross_attention3(x3, x_fft)
        x4 = self.cross_attention4(x4, x_fft)
        x_fft = self.cross_attention5(x_fft, x_cat)


        #Concatenate the outputs
        x = torch.cat([x, x2, x3, x4, x_fft], dim=1)

        #Conv1d to reduce the number of channels
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        #Pass to the lstm
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        
        # Get only the last value from the lstm output
        x = x[:, -1, :]

        x = self.fc(x)
        
        return x
        