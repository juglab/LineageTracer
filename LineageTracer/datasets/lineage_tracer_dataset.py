import glob
import numpy as np
import os
import random
import tifffile
from skimage.segmentation import relabel_sequential
from torch.utils.data import Dataset

from LineageTracer.utils.preprocess_data import normalize_min_max_percentile, load_pickle


class LineageTracerTrainValDataset(Dataset):
  def __init__(self, data_dir='./', type="train", size=None, num_sampled_tracklets=24,
               num_fg_points=500, std_object_size=20, bg_id=0, transform=None, ):

    print('`{}` dataloader created! Accessing data from {}/{}/'.format(type, data_dir, type))
    # get image and instance list
    dicts_list = glob.glob(os.path.join(data_dir, '{}/'.format(type), '*.pkl'))
    dicts_list.sort()
    print('Number of tracklets in `{}` directory is {}'.format(type, len(dicts_list)))

    self.dicts_list = dicts_list
    self.bg_id = bg_id
    self.size = size
    self.real_size = len(self.dicts_list)
    self.num_sampled_tracklets = num_sampled_tracklets
    self.num_fg_points = num_fg_points
    self.object_size = std_object_size
    self.transform = transform
    self.type = type

  def __len__(self):
    return self.real_size if self.size is None else self.size

  def __getitem__(self, index):

    indices = random.sample(range(self.real_size), self.num_sampled_tracklets)
    minibatch_tracklets = [load_pickle(self.dicts_list[index]) for index in
                           indices]  # list of size 24. Each member is a tracklet and is itself a list of dictionaries

    sample = {}
    sample['point_features'] = []  # 128 (latent) + offset (2/3) + intensity(1/2/3) properties for each point
    sample['normalized_global'] = []  # 4 properties for each point (2d) or 6 properties for each point (3d)
    sample['num_fg'] = []
    sample['labels'] = []

    for i, tracklet in enumerate(minibatch_tracklets):  # tracklet is a list of avatars of instance
      instance_id = i + 1
      instance_duration = len(tracklet)
      if instance_duration > 2:
        time_points = sorted(random.sample(list(np.arange(instance_duration)), 3))  # sampling without replacement!
      elif instance_duration >= 1:
        time_points = sorted(random.choices(list(np.arange(instance_duration)), k=3))
      frames = [tracklet[time_points[0]], tracklet[time_points[1]], tracklet[time_points[2]]]

      for frame in frames:  # list of three frames
        img_crop = frame['img_crop']  # H x W or D x H x W
        mask_crop = frame['mask_crop']  # H x W or D x H x W
        mask_crop_ = (mask_crop > 0)
        global_position = frame['global_position']
        image_dims = frame['image_dims']  # list [h, w] or [d, h, w]
        num_fg_pixels = int(self.num_fg_points)

        # get center of foreground
        if len(image_dims) == 2:
          y_fg, x_fg = np.where(mask_crop_ == 1)
          y_fg_mean, x_fg_mean = np.mean(y_fg), np.mean(x_fg)
          y_fg_norm_global = (y_fg + global_position[0]) / image_dims[0]  # btw 0 and 1
          x_fg_norm_global = (x_fg + global_position[1]) / image_dims[1]  # btw 0 and 1
          sample['normalized_global'].append(
            [x_fg_norm_global.min(), y_fg_norm_global.min(), x_fg_norm_global.max(), y_fg_norm_global.max()])
          y_fg_ = (y_fg - y_fg_mean) / (self.object_size)  # roughly equivalent to doing mean 0, std 1
          x_fg_ = (x_fg - x_fg_mean) / (self.object_size)  # roughly equivalent to doing mean 0, std 1
        elif len(image_dims) == 3:
          z_fg, y_fg, x_fg = np.where(mask_crop_ == 1)
          z_fg_mean, y_fg_mean, x_fg_mean = np.mean(z_fg), np.mean(y_fg), np.mean(x_fg)
          z_fg_norm_global = (z_fg + global_position[0]) / image_dims[0]  # btw 0 and 1
          y_fg_norm_global = (y_fg + global_position[1]) / image_dims[1]  # btw 0 and 1
          x_fg_norm_global = (x_fg + global_position[2]) / image_dims[2]  # btw 0 and 1
          sample['normalized_global'].append(
            [x_fg_norm_global.min(), y_fg_norm_global.min(), z_fg_norm_global.min(),
             x_fg_norm_global.max(), y_fg_norm_global.max(), z_fg_norm_global.max()])
          z_fg_ = (z_fg - z_fg_mean) / (self.object_size)  # roughly equivalent to doing mean 0, std 1
          y_fg_ = (y_fg - y_fg_mean) / (self.object_size)  # roughly equvalent to doing mean 0, std 1
          x_fg_ = (x_fg - x_fg_mean) / (self.object_size)  # roughly equivalent to doing mean 0, std 1
        fg_intensities = img_crop[mask_crop_]  # (N,)
        fg_intensities = fg_intensities[..., np.newaxis]  # (N, 1)

        if len(image_dims) == 2:
          point_fg_features = np.concatenate(
            [y_fg_[:, np.newaxis], x_fg_[:, np.newaxis], fg_intensities],
            axis=1)  # N x (2 + 1 + 128) --> y, x, intensity, latent
        elif len(image_dims) == 3:
          point_fg_features = np.concatenate(
            [z_fg_[:, np.newaxis], y_fg_[:, np.newaxis], x_fg_[:, np.newaxis], fg_intensities],
            axis=1)  # N x (3 + 1 + 128) --> z, y, x, intensity, latent

        choices = np.random.choice(point_fg_features.shape[0], num_fg_pixels)
        points_fg = point_fg_features[choices][np.newaxis, ...].astype(np.float32)  # ~1 x  100 x 131
        sample['point_features'].append(points_fg)
        sample['labels'].append(np.array(instance_id)[np.newaxis])
        sample['num_fg'].append(num_fg_pixels)
    sample['point_features'] = np.concatenate(sample['point_features'], axis=0)  # (72 x 100 x 131)
    sample['num_fg'] = np.array(sample['num_fg'], dtype=np.float32)  # (72, )
    #
    # [100. 100. ... 100 ]
    sample['labels'] = np.concatenate(sample['labels'], axis=0)  # (72, )
    # sample['labels]
    # [1  1  1  2  2  2  3  3  3  4  4  4  5  5  5  6  6  6  7  7  7  8  8  8
    #  9  9  9 10 10 10 11 11 11 12 12 12 13 13 13 14 14 14 15 15 15 16 16 16
    #  17 17 17 18 18 18 19 19 19 20 20 20 21 21 21 22 22 22 23 23 23 24 24 24]
    sample['normalized_global'] = np.array(sample['normalized_global'], dtype=np.float32)  # (72, 4)

    return sample


class LineageTracerTestDataset(Dataset):
  def __init__(self, data_dir='./', type="test", size=None, num_sampled_tracklets=24,
               num_fg_points=500, std_object_size=20, bg_id=0, transform=None, ):

    print('`{}` dataloader created! Accessing data from {}/{}/'.format(type, data_dir, type))
    # get image and instance list
    crops_list = glob.glob(os.path.join(data_dir, '{}/'.format(type), '*.pkl'))
    crops_list.sort()
    print('Number of tracklets in `{}` directory is {}'.format(type, len(crops_list)))

    self.crops_list = crops_list
    self.bg_id = bg_id
    self.size = size
    self.real_size = len(self.crops_list)
    self.num_sampled_tracklets = num_sampled_tracklets
    self.num_fg_points = num_fg_points
    self.object_size = std_object_size
    self.transform = transform
    self.type = type

  def __len__(self):
    return self.real_size if self.size is None else self.size

  def __getitem__(self, index):
    index = index if self.size is None else random.randint(0, self.real_size - 1)
    tp_ids_crops = load_pickle(self.crops_list[index])  # list of object dictionaries at a given time point
    sequence = os.path.basename(os.path.dirname(self.crops_list[index]))

    sample = {}
    sample['point_features'] = []  # 128 (latent) + offset (2/3) + intensity(1/2/3) properties for each point
    sample['normalized_global'] = []  # 4 properties for each point (2d) or 6 properties for each point (3d)
    sample['num_fg'] = []
    sample['time_point'] = []
    sample['object_id'] = []
    sample['sequence'] = []

    for object_data in tp_ids_crops:  # list of object dictionaries pertaining to this time point

      img_crop = object_data['img_crop']  # H x W
      mask_crop = object_data['mask_crop']  # H x W
      mask_crop_ = (mask_crop > 0)
      global_position = object_data['global_position']
      image_dims = object_data['image_dims']  # list [h, w]
      num_fg_pixels = int(self.num_fg_points)

      # get center of foreground
      if len(image_dims) == 2:
        y_fg, x_fg = np.where(mask_crop_ == 1)
        y_fg_mean, x_fg_mean = np.mean(y_fg), np.mean(x_fg)
        y_fg_norm_global = (y_fg + global_position[0]) / image_dims[0]
        x_fg_norm_global = (x_fg + global_position[1]) / image_dims[1]
        sample['normalized_global'].append(
          [x_fg_norm_global.min(), y_fg_norm_global.min(), x_fg_norm_global.max(), y_fg_norm_global.max()])
        y_fg_ = (y_fg - y_fg_mean) / self.object_size
        x_fg_ = (x_fg - x_fg_mean) / self.object_size
      elif len(image_dims) == 3:
        z_fg, y_fg, x_fg = np.where(mask_crop_ == 1)
        z_fg_mean, y_fg_mean, x_fg_mean = np.mean(z_fg), np.mean(y_fg), np.mean(x_fg)
        z_fg_norm_global = (z_fg + global_position[0]) / image_dims[0]
        y_fg_norm_global = (y_fg + global_position[1]) / image_dims[1]
        x_fg_norm_global = (x_fg + global_position[2]) / image_dims[2]

        sample['normalized_global'].append(
          [x_fg_norm_global.min(), y_fg_norm_global.min(), z_fg_norm_global.min(), x_fg_norm_global.max(),
           y_fg_norm_global.max(), z_fg_norm_global.max()])
        z_fg_ = (z_fg - z_fg_mean) / (self.object_size)
        y_fg_ = (y_fg - y_fg_mean) / (self.object_size)
        x_fg_ = (x_fg - x_fg_mean) / (self.object_size)

      fg_intensities = img_crop[mask_crop_]  # (N,)
      fg_intensities = fg_intensities[..., np.newaxis]  # (N, 1)

      if len(image_dims) == 2:
        point_fg_features = np.concatenate(
          [y_fg_[:, np.newaxis], x_fg_[:, np.newaxis], fg_intensities],
          axis=1)  # N x (3) --> y, x, intensity, latent
      elif len(image_dims) == 3:
        point_fg_features = np.concatenate(
          [z_fg_[:, np.newaxis], y_fg_[:, np.newaxis], x_fg_[:, np.newaxis], fg_intensities],
          axis=1)
      choices = np.random.choice(point_fg_features.shape[0], num_fg_pixels)
      points_fg = point_fg_features[choices][np.newaxis, ...].astype(np.float32)  # ~1 x  100 x 131

      sample['point_features'].append(points_fg)
      sample['num_fg'].append(num_fg_pixels)
      sample['object_id'].append(object_data['id'])
      sample['time_point'].append(str(index))  # TODO

    if len(tp_ids_crops) > 0:
      sample['point_features'] = np.concatenate(sample['point_features'], axis=0)  # (1 object x 100 x 131)
      sample['num_fg'] = np.array(sample['num_fg'], dtype=np.float32)  # (1 object, )
      sample['object_id'] = np.array(sample['object_id'], dtype=np.int16)  # (1 object, )
      sample['time_point'] = np.array(sample['time_point'], dtype=np.int16)  # (1 object, )
      sample['normalized_global'] = np.array(sample['normalized_global'], dtype=np.float32)  # (1 object, 4)
      sample['sequence'] = sequence
    return sample
