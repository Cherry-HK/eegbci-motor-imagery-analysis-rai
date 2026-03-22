import math

import torch
import torch.nn as nn


class EEGNetLite(nn.Module):
    def __init__(
        self,
        n_channels,
        n_samples,
        temporal_filters=8,
        depth_multiplier=2,
        kernel_length=64,
        dropout_rate=0.5,
    ):
        super().__init__()

        spatial_filters = temporal_filters * depth_multiplier
        kernel_length = min(kernel_length, n_samples)
        padding = kernel_length // 2

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                temporal_filters,
                kernel_size=(1, kernel_length),
                padding=(0, padding),
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                kernel_size=(n_channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, spatial_filters, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            flattened_dim = self.features(dummy).reshape(1, -1).shape[1]

        self.classifier = nn.Linear(flattened_dim, 2)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


class EEGShallowConvNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_samples,
        temporal_filters=40,
        filter_length=25,
        pool_length=75,
        pool_stride=15,
        dropout_rate=0.5,
    ):
        super().__init__()

        filter_length = max(3, min(filter_length, n_samples))
        temporal_output_samples = max(1, n_samples - filter_length + 1)
        pool_length = max(2, min(pool_length, temporal_output_samples))
        pool_stride = max(1, min(pool_stride, pool_length))

        self.temporal_conv = nn.Conv2d(
            1,
            temporal_filters,
            kernel_size=(1, filter_length),
            bias=False,
        )
        self.spatial_conv = nn.Conv2d(
            temporal_filters,
            temporal_filters,
            kernel_size=(n_channels, 1),
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(temporal_filters)
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_length), stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout_rate)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            features = self._forward_features(dummy)
            flattened_dim = features.reshape(1, -1).shape[1]

        self.classifier = nn.Linear(flattened_dim, 2)

    def _forward_features(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.batch_norm(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


class EEGLSTM(nn.Module):
    def __init__(
        self,
        n_channels,
        hidden_size=64,
        num_layers=1,
        dropout_rate=0.3,
        bidirectional=False,
    ):
        super().__init__()

        self.num_directions = 2 if bidirectional else 1
        effective_dropout = dropout_rate if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size * self.num_directions, 2)

    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        last_step = self.dropout(last_step)
        return self.classifier(last_step)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class EEGTransformer(nn.Module):
    def __init__(
        self,
        n_channels,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(n_channels, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)
