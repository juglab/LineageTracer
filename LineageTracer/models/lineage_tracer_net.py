import torch
import torch.nn as nn
import torch.nn.functional as F


class LineageTracerNet(nn.Module):
  def __init__(self,
               num_fg_points=500,
               num_offset_channels=2,  # Y X
               num_intensity_channels=1,
               num_latent_channels=128,
               num_output_channels=32,
               ):
    super().__init__()
    self.ap1 = nn.AvgPool1d(num_fg_points)
    self.mp2 = nn.MaxPool1d(num_fg_points)

    self.num_intensity_channels = num_intensity_channels
    self.num_offset_channels = num_offset_channels
    self.num_latent_channels = num_latent_channels

    if self.num_offset_channels > 0:
      self.offset_conv1 = nn.Conv1d(num_offset_channels, 64, 1)
      self.offset_conv2 = nn.Conv1d(64, 128, 1)
      self.offset_conv3 = nn.Conv1d(128, 256, 1)
      self.offset_conv1_bn = nn.BatchNorm1d(64)
      self.offset_conv2_bn = nn.BatchNorm1d(128)
      self.offset_conv3_bn = nn.BatchNorm1d(256)

    if self.num_intensity_channels > 0:
      self.intensity_conv1 = nn.Conv1d(num_intensity_channels, 64, 1)
      self.intensity_conv2 = nn.Conv1d(64, 128, 1)
      self.intensity_conv3 = nn.Conv1d(128, 256, 1)
      self.intensity_conv1_bn = nn.BatchNorm1d(64)
      self.intensity_conv2_bn = nn.BatchNorm1d(128)
      self.intensity_conv3_bn = nn.BatchNorm1d(256)

    if self.num_latent_channels > 0:
      self.latent_conv1 = nn.Conv1d(num_latent_channels, 64, 1)
      self.latent_conv2 = nn.Conv1d(64, 128, 1)
      self.latent_conv3 = nn.Conv1d(128, 256, 1)
      self.latent_conv1_bn = nn.BatchNorm1d(64)
      self.latent_conv2_bn = nn.BatchNorm1d(128)
      self.latent_conv3_bn = nn.BatchNorm1d(256)

    if self.num_offset_channels > 0 and self.num_intensity_channels > 0 and self.num_latent_channels > 0:
      self.conv4 = nn.Conv1d(256 * 3, 256, 1)
      if self.num_offset_channels == 2:
        self.last_layer = nn.Sequential(
          nn.Linear(128 + 256 * 3 + 64, 256),
          nn.LeakyReLU(),
          nn.Linear(256, num_output_channels))
      elif self.num_offset_channels == 3:
        self.last_layer = nn.Sequential(
          nn.Linear(128 + 256 * 3 + 96, 256),
          nn.LeakyReLU(),
          nn.Linear(256, num_output_channels))
    elif self.num_offset_channels > 0 and self.num_intensity_channels > 0 and self.num_latent_channels == 0:
      self.conv4 = nn.Conv1d(256 * 2, 256, 1)
      if self.num_offset_channels == 2:
        self.last_layer = nn.Sequential(
          nn.Linear(128 + 256 * 2 + 64, 256),
          nn.LeakyReLU(),
          nn.Linear(256, num_output_channels))
      elif self.num_offset_channels == 3:
        self.last_layer = nn.Sequential(
          nn.Linear(128 + 256 * 2 + 96, 256),
          nn.LeakyReLU(),
          nn.Linear(256, num_output_channels))
    elif self.num_offset_channels > 0 and self.num_intensity_channels == 0 and self.num_latent_channels == 0:
      self.conv4 = nn.Conv1d(256, 256, 1)
      if self.num_offset_channels == 2:
        self.last_layer = nn.Sequential(
          nn.Linear(128 + 256 + 64, 256),
          nn.LeakyReLU(),
          nn.Linear(256, num_output_channels))
      elif self.num_offset_channels == 3:
        self.last_layer = nn.Sequential(
          nn.Linear(128 + 256 + 96, 256),
          nn.LeakyReLU(),
          nn.Linear(256, num_output_channels))
    self.conv5 = nn.Conv1d(256, 512, 1)
    self.conv6 = nn.Conv1d(512, 64, 1)
    self.conv4_bn = nn.BatchNorm1d(256)
    self.conv5_bn = nn.BatchNorm1d(512)
    self.conv7 = nn.Conv1d(512, 256, 1)
    self.conv8 = nn.Conv1d(256, 512, 1)
    self.conv9 = nn.Conv1d(512, 64, 1)

    self.conv7_bn = nn.BatchNorm1d(256)
    self.conv8_bn = nn.BatchNorm1d(512)

    self.conv_weight = nn.Conv1d(128, 1, 1)

  def location_embedding(self, f_g, dim_g=64, wave_len=1000):
    # this produces a 96 dimensional representation of each expanded bounding box!
    # f_g has shape ~72 x 4
    """

    :param f_g: 72 x 4
    :param dim_g:
    :param wave_len:
    :return:
    """
    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=1)  # ~72 x 1 after chunking

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    position_mat = torch.cat((cx, cy, w, h), -1)
    feat_range = torch.arange(dim_g / 8).cuda()
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, 1, -1)  # (1, 1, 8)
    position_mat = position_mat.view(f_g.shape[0], 4, -1)
    position_mat = 100. * position_mat  # ~72 x 4 x 1

    mul_mat = position_mat * dim_mat  # ~72 x 4 x 8
    mul_mat = mul_mat.view(f_g.shape[0], -1)  # ~ 72 x 32
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)  # 72 x 64
    return embedding

  def location_embedding_3d(self, f_g, dim_g=64, wave_len=1000):
    # this produces a 96 dimensional representation of each expanded bounding box!
    # f_g has shape ~72 x 6
    """

    :param f_g: 72 x 6
    :param dim_g:
    :param wave_len:
    :return:
    """
    x_min, y_min, z_min, x_max, y_max, z_max = torch.chunk(f_g, 6, dim=1)  # each is ~72 x 1 after chunking

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    cz = (z_min + z_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    d = (z_max - z_min) + 1.
    position_mat = torch.cat((cx, cy, cz, w, h, d), -1)  # ~72 x 6
    feat_range = torch.arange(dim_g / 8).cuda()
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))
    dim_mat = dim_mat.view(1, 1, -1)  # (1, 1, 8)
    position_mat = position_mat.view(f_g.shape[0], 6, -1)
    position_mat = 100. * position_mat  # ~72 x 6 x 1

    mul_mat = position_mat * dim_mat  # ~72 x 6 x 8
    mul_mat = mul_mat.view(f_g.shape[0], -1)  # ~ 72 x 48
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)  # 72 x 96
    return embedding

  def init_output(self):
    with torch.no_grad():
      pass

  def forward(self, point_features, normalized_global_xyz):
    """
    Parameters
    ----------
    point_features: (1, 72, 100, 131)

    normalized_global_xyz: (1, 72, 4) or (1, 72, 6)

    Returns
    ----------

    """

    point_features, normalized_global_xyz = point_features[0], normalized_global_xyz[0]  # remove batch dimension
    if normalized_global_xyz.shape[-1] == 4:
      spatial_embs = self.location_embedding(normalized_global_xyz)
    elif normalized_global_xyz.shape[-1] == 6:
      spatial_embs = self.location_embedding_3d(normalized_global_xyz)

    point_features = point_features.transpose(2, 1).contiguous()
    offsets, intensities = point_features[:, :self.num_offset_channels], point_features[:,
                                                                         self.num_offset_channels:self.num_offset_channels + self.num_intensity_channels],

    offsets = F.leaky_relu(self.offset_conv1_bn(self.offset_conv1(offsets)))
    intensities = F.leaky_relu(self.intensity_conv1_bn(self.intensity_conv1(intensities)))

    offsets = F.leaky_relu(self.offset_conv2_bn(self.offset_conv2(offsets)))
    intensities = F.leaky_relu(self.intensity_conv2_bn(self.intensity_conv2(intensities)))

    offsets = F.leaky_relu(self.offset_conv3_bn(self.offset_conv3(offsets)))  # 72 x 256 x 100
    intensities = F.leaky_relu(self.intensity_conv3_bn(self.intensity_conv3(intensities)))  # 72 x 256 x 100

    point_feat_2 = torch.cat((offsets, intensities), dim=1)  # 72 x 512 x 100

    x1 = F.leaky_relu(self.conv4_bn(self.conv4(point_feat_2)))
    x1 = F.leaky_relu(self.conv5_bn(self.conv5(x1)))
    x1 = F.leaky_relu(self.conv6(x1))  # 72 x 64 x 100
    ap_x1 = self.ap1(x1).squeeze(-1)  # 72 x 64 (Average Pool)

    x2 = F.leaky_relu(self.conv7_bn(self.conv7(point_feat_2)))
    x2 = F.leaky_relu(self.conv8_bn(self.conv8(x2)))
    x2 = F.leaky_relu(self.conv9(x2))  # 72 x 64 x 100
    mp_x2 = self.mp2(x2).squeeze(-1)  # 72 x 64 (Max Pool)

    weight_feat = self.conv_weight(torch.cat([x1, x2], dim=1))  # 72 x 1 x 100
    weight = torch.nn.Softmax(2)(weight_feat)
    weight_x3 = (weight.expand_as(point_feat_2) * point_feat_2).sum(2)  # 72 x 512
    offsets = torch.cat([ap_x1, mp_x2, weight_x3, spatial_embs], dim=1)  # 72 X (64+64+512+96)
    instance_embeddings = self.last_layer(offsets)

    return instance_embeddings
