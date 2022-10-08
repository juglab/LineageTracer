import os
import torch


def create_test_configs_dict(crops_dir,
                             data_dir,
                             project_name,
                             checkpoint_path,
                             num_sampled_tracklets,
                             mean_tracklet_length,
                             min_tracklet_length,
                             save_dir=None,
                             num_fg_points=500,
                             std_object_size=20,
                             save_images=True,
                             save_results=True,
                             cuda=True,
                             num_workers=4,
                             type='test',
                             bg_id=0,
                             batch_size=1,
                             num_offset_channels=2,
                             num_intensity_channels=1,
                             num_latent_channels=0,
                             num_output_channels=32,
                             margin=0.2,
                             ):
  """
      Creates `test_configs` dictionary from parameters.
      Parameters
      ----------
      crops_dir: str
          Crops from the GT masks have been previously saved as *.pkl objects.
          Pixels sampled from these are fed into the network.
      data_dir : str
          Data is read from os.path.join(data_dir, 'test')
      checkpoint_path: str
          This indicates the path to the trained model
      data_type: str
          This reflects the data-type of the image and should be equal to one of '8-bit' or '16-bit'
      save_dir: str
          This indicates the directory where the results are saved
      save_images: boolean
          If True, then prediction images are saved
      save_results: boolean
          If True, then prediction results are saved in text file
      cuda: boolean
          True, indicates GPU usage
      type: str
          One of 'train', 'val' or 'test'
      bg_id : int, default = 0
          Label of background in the ground truth test label masks
          This parameter is only used while quantifying accuracy predicted label masks with ground truth label masks
      num_workers: int, default = 4
  """

  test_configs = dict(
    cuda=cuda,
    save_results=save_results,
    save_images=save_images,
    save_dir=save_dir,
    checkpoint_path=checkpoint_path,
    mean_tracklet_length=mean_tracklet_length,
    min_tracklet_length=min_tracklet_length,
    batch_size=batch_size,
    num_workers=num_workers,
    project_name=project_name,
    data_dir=os.path.join(data_dir, project_name),
    dataset={
      'type': type,
      'kwargs': {
        'data_dir': crops_dir,
        'type': type,
        'bg_id': bg_id,
        'num_sampled_tracklets': num_sampled_tracklets,
        'num_fg_points': num_fg_points,
        'std_object_size': std_object_size,
      },
    },
    model={
      'name': 'tracker_net',
      'kwargs': {
        'num_fg_points': num_fg_points,
        'num_offset_channels': num_offset_channels,
        'num_intensity_channels': num_intensity_channels,
        'num_latent_channels': num_latent_channels,
        'num_output_channels': num_output_channels
      }
    },
    loss_dict={
      'margin': margin
    }

  )
  print(
    "`test_configs` dictionary successfully created with: "
    "\n -- evaluation images accessed from {}, "
    "\n -- trained weights accessed from {}, "
    "\n -- output directory chosen as {}".format(
      data_dir, checkpoint_path, save_dir))
  return test_configs


def create_dataset_dict(data_dir,
                        project_name,
                        size,
                        type,
                        num_sampled_tracklets=24,
                        num_fg_points=500,
                        std_object_size=20,
                        name='2d',
                        batch_size=1,
                        workers=8,
                        ):
  """
      Creates `dataset_dict` dictionary from parameters.
      Parameters
      ----------
      data_dir: string
          Data is read from os.path.join(data_dir, project_name)
      project_name: string
          Data is read from os.path.join(data_dir, project_name)
      size: int
          Number of image-mask per epoch
      type: string
          One of 'train', 'val'
      num_sampled_tracklets: int
          Number of tracklets sampled per step of gradient update
          Default = 24
      num_fg_points: int
          Number of pixels (voxels) sampled per object instance
          Default = 500
      std_object_size: float
          Spread of pixels around object center
      name: string
          One of '2d' or '3d'
      batch_size: int
          Effective Batch-size is the product of `batch_size` and `virtual_batch_multiplier`
      workers: int
          Number of data-loader workers
  """
  dataset_dict = {
    'type': type,
    'kwargs': {
      'data_dir': os.path.join(data_dir),
      'type': type,
      'size': size,
      'num_sampled_tracklets': num_sampled_tracklets,
      'num_fg_points': num_fg_points,
      'std_object_size': std_object_size,
    },
    'batch_size': batch_size,
    'workers': workers,

  }
  print("`{}_dataset_dict` dictionary successfully created with: \n -- {} images accessed from {}, "
        "\n -- number of images per epoch equal to {}, "
        "\n -- batch size set at {}, "
        .format(type, type, os.path.join(data_dir, project_name, type, 'images'), size, batch_size,
                ))
  return dataset_dict


def create_model_dict(num_fg_points,
                      num_offset_channels=2,
                      num_intensity_channels=1,
                      num_latent_channels=0,
                      num_output_channels=32):
  """
      Creates `model_dict` dictionary from parameters.

      Parameters
      ----------
      num_fg_points : int
                  Number of pixels sampled per object instance
      num_offset_channels: int
                  Default: 2 for 2D, 3 for 3D
      num_intensity_channels: int
                  Default: 1
      num_latent_channels: int
                  Dimensionality of latent dimension
      num_output_channels: int
                  Dimensionality of output embedding

      Returns
      ----------
      model_dict: Dictionary
  """
  model_dict = {
    'name': 'tracker_net',
    'kwargs': {
      'num_fg_points': num_fg_points,
      'num_offset_channels': num_offset_channels,
      'num_intensity_channels': num_intensity_channels,
      'num_latent_channels': num_latent_channels,
      'num_output_channels': num_output_channels
    }
  }
  print(
    "`model_dict` dictionary successfully created with: \n -- number of offset channels equal to {}, "
    "\n -- number of intensity channels equal to {}, "
    "\n -- number of latent channels equal to {}, "
    "\n -- number of output channels equal to {}".format(
      num_offset_channels, num_intensity_channels, num_latent_channels, num_output_channels))
  return model_dict


def create_configs(save_dir,
                   resume_path,
                   n_epochs=200,
                   train_lr=5e-4,
                   cuda=True,
                   save=True,
                   save_checkpoint_frequency=None,
                   ):
  """
      Creates `configs` dictionary from parameters.
      Parameters
      ----------
      save_dir: str
          Path to where the experiment is saved
      resume_path: str
          Path to where the trained model (for example, checkpoint.pth) lives
      n_epochs: int
          Total number of epochs
      train_lr: float
          Starting learning rate
      cuda: boolean
          If True, use GPU
      save: boolean
          If True, then results are saved
      save_checkpoint_frequency: int
          Save model weights after 'n' epochs (in addition to last and best model weights)
          Default is None
  """
  configs = dict(train_lr=train_lr,
                 n_epochs=n_epochs,
                 cuda=cuda,
                 save=save,
                 save_dir=save_dir,
                 resume_path=resume_path,
                 save_checkpoint_frequency=save_checkpoint_frequency
                 )
  print(
    "`configs` dictionary successfully created with: "
    "\n -- n_epochs equal to {}, "
    "\n -- save_dir equal to {}, "
      .format(n_epochs, save_dir))
  return configs


def create_loss_dict(margin=0.2):
  """
      margin: float
          Default = 0.2
  """
  loss_dict = {
    'margin': margin
  }
  print(
    "`loss_dict` dictionary successfully created with: \n -- margin equal to {:.3f}".format(
      margin))
  return loss_dict
