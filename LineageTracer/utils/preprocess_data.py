import numpy as np
import os
import pickle
import shutil
import subprocess as sp
import tifffile
import urllib.request
import zipfile
from glob import glob
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def extract_data(zip_url, project_name, data_dir='../../../data/'):
  """
      Extracts data from `zip_url` to the location identified by `data_dir` and `project_name` parameters.

      Parameters
      ----------
      zip_url: string
          Indicates the external url
      project_name: string
          Indicates the path to the sub-directory at the location identified by the parameter `data_dir`
      data_dir: string
          Indicates the path to the directory where the data should be saved.
      Returns
      -------

  """
  zip_path = os.path.join(data_dir, project_name + '.zip')

  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print("Created new directory {}".format(data_dir))

  if (os.path.exists(zip_path)):
    print("Zip file was downloaded and extracted before!")
  else:
    if (os.path.exists(os.path.join(data_dir, project_name, 'download/'))):
      pass
    else:
      os.makedirs(os.path.join(data_dir, project_name, 'download/'))
      urllib.request.urlretrieve(zip_url, zip_path)
      print("Downloaded data as {}".format(zip_path))
      with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
      if os.path.exists(os.path.join(data_dir, os.path.basename(zip_url)[:-4], 'train')):
        shutil.move(os.path.join(data_dir, os.path.basename(zip_url)[:-4], 'train'),
                    os.path.join(data_dir, project_name, 'download/'))
      if os.path.exists(os.path.join(data_dir, os.path.basename(zip_url)[:-4], 'val')):
        shutil.move(os.path.join(data_dir, os.path.basename(zip_url)[:-4], 'val'),
                    os.path.join(data_dir, project_name, 'download/'))
      if os.path.exists(os.path.join(data_dir, os.path.basename(zip_url)[:-4], 'test')):
        shutil.move(os.path.join(data_dir, os.path.basename(zip_url)[:-4], 'test'),
                    os.path.join(data_dir, project_name, 'download/'))
      print("Unzipped data to {}".format(os.path.join(data_dir, project_name, 'download/')))


def reid_masks(data_dir, project_name):
  """

  Re-assigns ids to the instance segmentation files in directory `masks` to be consistent with the
  GT tracking annotations directory `tracking_annotations`

  Parameters
  ----------
  data_dir: string
      Indicates the path where all the data is present (for example, `../../../data`)
  project_name: string
      Indicates the name of the project (for example, `Fluo-N2DH-GOWT1`)
  Returns
  ----------
  """
  instance_path_train = os.path.join(data_dir, project_name, 'train', 'masks/')
  track_path_train = os.path.join(data_dir, project_name, 'train', 'tracking-annotations/')

  instance_names_train = sorted(glob(os.path.join(instance_path_train, '*.tif')))
  track_names_train = sorted(glob(os.path.join(track_path_train, '*.tif')))

  instance_path_train_reid = os.path.join(data_dir, project_name, 'train', 'masks-reid/')
  if not os.path.exists(instance_path_train_reid):
    os.makedirs(os.path.dirname(instance_path_train_reid))
    print("Created new directory : {}".format(instance_path_train_reid))

  for k in tqdm(range(len(instance_names_train))):
    instance_train = tifffile.imread(instance_names_train[k])
    track_train = tifffile.imread(track_names_train[k])
    instance_train_reid = np.zeros_like(instance_train)
    ids_instance = np.unique(instance_train)
    ids_instance = ids_instance[ids_instance != 0]

    ids_track = np.unique(track_train)
    ids_track = ids_track[ids_track != 0]

    IoU_table = np.zeros((len(ids_track), len(ids_instance)))

    for i, id_track in enumerate(ids_track):
      for j, id_instance in enumerate(ids_instance):

        intersection = ((track_train == id_track)
                        & (instance_train == id_instance)).sum()
        union = ((track_train == id_track)
                 | (instance_train == id_instance)).sum()
        if union != 0:
          IoU_table[i, j] = intersection / union
        else:
          IoU_table[i, j] = 0.0
    row_indices, col_indices = linear_sum_assignment(-IoU_table)
    matched_indices = np.array(list(zip(row_indices, col_indices)))  # list of (row, col) tuples
    for m in matched_indices:
      if (IoU_table[m[0], m[1]] > 0):
        instance_train_reid[instance_train == ids_instance[m[1]]] = ids_track[m[0]]
    tifffile.imsave(instance_path_train_reid + '/' + os.path.basename(instance_names_train[k]), instance_train_reid)


def make_dirs(data_dir, project_name):
  """
      Makes directories - `train`, `val, `test` and subdirectories under each `images` and `masks`

      Parameters
      ----------
      data_dir: string
          Indicates the path where the `project` lives.
      project_name: string
          Indicates the name of the sub-folder under the location identified by `data_dir`.

      Returns
      -------
  """
  image_path_train = os.path.join(data_dir, project_name, 'train', 'images/')
  instance_path_train = os.path.join(data_dir, project_name, 'train', 'masks/')

  image_path_val = os.path.join(data_dir, project_name, 'val', 'images/')
  instance_path_val = os.path.join(data_dir, project_name, 'val', 'masks/')

  image_path_test = os.path.join(data_dir, project_name, 'test', 'images/')
  instance_path_test = os.path.join(data_dir, project_name, 'test', 'masks/')
  track_path_test = os.path.join(data_dir, project_name, 'test', 'tracking-annotations/')

  if not os.path.exists(image_path_train):
    os.makedirs(os.path.dirname(image_path_train))
    print("Created new directory : {}".format(image_path_train))

  if not os.path.exists(instance_path_train):
    os.makedirs(os.path.dirname(instance_path_train))
    print("Created new directory : {}".format(instance_path_train))

  if not os.path.exists(image_path_val):
    os.makedirs(os.path.dirname(image_path_val))
    print("Created new directory : {}".format(image_path_val))

  if not os.path.exists(instance_path_val):
    os.makedirs(os.path.dirname(instance_path_val))
    print("Created new directory : {}".format(instance_path_val))

  if not os.path.exists(image_path_test):
    os.makedirs(os.path.dirname(image_path_test))
    print("Created new directory : {}".format(image_path_test))

  if not os.path.exists(instance_path_test):
    os.makedirs(os.path.dirname(instance_path_test))
    print("Created new directory : {}".format(instance_path_test))

  if not os.path.exists(track_path_test):
    os.makedirs(os.path.dirname(track_path_test))
    print("Created new directory : {}".format(track_path_test))


def split_train_val(data_dir, project_name, train_val_name, subset=0.15, seed=1000):
  """
      Splits the `train` directory into `val` directory using the partition percentage of `subset`.

      Parameters
      ----------
      data_dir: string
          Indicates the path where the `project` lives.
      project_name: string
          Indicates the name of the sub-folder under the location identified by `data_dir`.
      train_val_name: string
          Indicates the name of the sub-directory under `project-name` which must be split
      subset: float
          Indicates the fraction of data to be reserved for validation
      seed: integer
          Allows for the same partition to be used in each experiment.
          Change this if you would like to obtain results with different train-val partitions.
      Returns
      -------

  """

  image_dir = os.path.join(data_dir, project_name, 'download', train_val_name, 'images')
  instance_dir = os.path.join(data_dir, project_name, 'download', train_val_name, 'masks-reid')

  image_names = sorted(glob(os.path.join(image_dir, '*.tif')))
  instance_names = sorted(glob(os.path.join(instance_dir, '*.tif')))
  indices = np.arange(len(image_names))
  np.random.seed(seed)

  outside = True
  while outside:
    start_index = np.random.randint(len(image_names))
    end_index = start_index + int(subset * len(image_names)) + 1
    if end_index < len(image_names):
      outside = False

  val_indices = indices[start_index:end_index]

  train_indices_1 = indices[:start_index]
  train_indices_2 = indices[end_index:]
  train_indices = np.concatenate([train_indices_1, train_indices_2])

  make_dirs(data_dir=data_dir, project_name=project_name)

  for val_index in val_indices:
    shutil.copy(image_names[val_index], os.path.join(data_dir, project_name, 'val', 'images'))
    shutil.copy(instance_names[val_index], os.path.join(data_dir, project_name, 'val', 'masks'))

  for train_index in train_indices:
    shutil.copy(image_names[train_index], os.path.join(data_dir, project_name, 'train', 'images'))
    shutil.copy(instance_names[train_index], os.path.join(data_dir, project_name, 'train', 'masks'))

  image_dir = os.path.join(data_dir, project_name, 'download', 'test', 'images')
  instance_dir = os.path.join(data_dir, project_name, 'download', 'test', 'masks')
  track_dir = os.path.join(data_dir, project_name, 'download', 'test', 'tracking-annotations')
  image_names = sorted(glob(os.path.join(image_dir, '*.tif')))
  instance_names = sorted(glob(os.path.join(instance_dir, '*.tif')))
  track_names = sorted(glob(os.path.join(track_dir, '*.tif')))
  test_indices = np.arange(len(image_names))

  for test_index in test_indices:
    shutil.copy(image_names[test_index], os.path.join(data_dir, project_name, 'test', 'images'))
    shutil.copy(instance_names[test_index], os.path.join(data_dir, project_name, 'test', 'masks'))
    shutil.copy(track_names[test_index], os.path.join(data_dir, project_name, 'test', 'tracking-annotations'))

  shutil.copy(os.path.join(track_dir, 'man_track.txt'),
              os.path.join(data_dir, project_name, 'test', 'tracking-annotations'))
  print("Train-Val-Test Images/Masks copied to {}".format(os.path.join(data_dir, project_name)))


def normalize_min_max_percentile(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
  """
      Percentile-based image normalization.
      Function taken from StarDist repository  https://github.com/stardist/stardist
  """
  mi = np.percentile(x, pmin, axis=axis, keepdims=True)
  ma = np.percentile(x, pmax, axis=axis, keepdims=True)
  return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
  """
      Percentile-based image normalization.
      Function taken from StarDist repository  https://github.com/stardist/stardist
  """
  if dtype is not None:
    x = x.astype(dtype, copy=False)
    mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
    ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
    eps = dtype(eps)

  try:
    import numexpr
    x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
  except ImportError:
    x = (x - mi) / (ma - mi + eps)

  if clip:
    x = np.clip(x, 0, 1)

  return x


def save_pickle(filename, obj):
  """
  `save_pickle` pickles a list
  Parameters
  ----------
  filename: str
      Name of pickle object
  obj: list
  Returns
  ----------

  """
  with open(filename, 'wb') as f:
    pickle.dump(obj, f, protocol=2)


def load_pickle(filename):
  """
    `load_pickle` un-pickles a pickle object and returns a list
    Parameters
    ----------
    filename: str
        Name of pickle object

    Returns
    ----------
    obj: list
    """
  with open(filename, 'rb') as f:
    obj = pickle.load(f)
  return obj


def create_crops_3d(input_dir, train_val_name, crops_dir='./crops'):
  """
    `create_crops_3d` generates a list `instance_list_tp` for each time point where the entries in the list
    are individual dictionaries corresponding to each id

    Parameters
    ----------
    input_dir : str
          Path to images and masks
    train_val_name: str
          one of 'train' or 'val'
    crops_dir: str
          for example, this could be 'crops'

    Returns
    ----------
  """
  image_dir = os.path.join(input_dir, train_val_name, 'images')
  mask_dir = os.path.join(input_dir, train_val_name, 'masks')

  image_list = sorted(glob(image_dir + '/*.tif'))
  mask_list = sorted(glob(mask_dir + '/*.tif'))

  crops_dir = os.path.join(crops_dir, train_val_name)

  if not os.path.exists(crops_dir):
    os.makedirs(os.path.dirname(crops_dir))
    print("Created new directory : {}".format(crops_dir))

  for i in tqdm(range(len(image_list))):
    image = tifffile.imread(image_list[i])
    image = normalize_min_max_percentile(image, 1, 99.8, (0, 1, 2))
    mask = tifffile.imread(mask_list[i])
    d, h, w = mask.shape

    ids = np.unique(mask)
    ids = ids[ids != 0]
    instance_list_tp = []
    for id in ids:

      z, y, x = np.where(mask == id)
      zmin, zmax, ymin, ymax, xmin, xmax = np.min(z), np.max(z), np.min(y), np.max(y), np.min(x), np.max(x)
      zmin_ = int(np.maximum(0, zmin))
      zmax_ = int(np.minimum(d - 1, zmax))
      ymin_ = int(np.maximum(0, ymin))
      ymax_ = int(np.minimum(h - 1, ymax))
      xmin_ = int(np.maximum(0, xmin))
      xmax_ = int(np.minimum(w - 1, xmax))
      img_crop = image[zmin_:zmax_ + 1, ymin_:ymax_ + 1, xmin_:xmax_ + 1]  # image
      mask_crop = mask[zmin_:zmax_ + 1, ymin_:ymax_ + 1, xmin_:xmax_ + 1]  # instance
      class_label_crop = (mask > 0)[zmin_:zmax_ + 1, ymin_:ymax_ + 1, xmin_:xmax_ + 1]  # foreground
      instance_list_tp.append(
        {'id': id, 'global_position': [zmin, ymin, xmin], 'img_crop': img_crop, 'mask_crop': mask_crop,
         'class_label_crop': class_label_crop, 'image_dims': [d, h, w]})
    save_pickle(os.path.join(crops_dir, os.path.basename(image_list[i])[:-4] + '.pkl'), instance_list_tp)


def create_crops(input_dir, train_val_name, crops_dir='./crops'):
  """
  `create_crops` generates a list `instance_list_tp` for each time point where the entries in the list
  are individual dictionaries corresponding to each id

  Parameters
  ----------
  input_dir : str
        Path to images and masks
  train_val_name: str
        one of 'train' or 'val'
  crops_dir: str
        for example, this could be 'crops'

  Returns
  ----------

  """
  image_dir = os.path.join(input_dir, train_val_name, 'images')
  mask_dir = os.path.join(input_dir, train_val_name, 'masks')

  image_list = sorted(glob(image_dir + '/*.tif'))
  mask_list = sorted(glob(mask_dir + '/*.tif'))

  crops_dir = os.path.join(crops_dir, train_val_name)

  if not os.path.exists(crops_dir):
    os.makedirs(os.path.dirname(crops_dir))
    print("Created new directory : {}".format(crops_dir))

  for i in tqdm(range(len(image_list))):
    image = tifffile.imread(image_list[i])
    image = normalize_min_max_percentile(image, 1, 99.8, (0, 1))
    mask = tifffile.imread(mask_list[i])
    h, w = mask.shape

    ids = np.unique(mask)
    ids = ids[ids != 0]
    instance_list_tp = []
    for id in ids:
      y, x = np.where(mask == id)
      ymin, ymax, xmin, xmax = np.min(y), np.max(y), np.min(x), np.max(x)
      ymin_ = int(np.maximum(0, ymin))
      ymax_ = int(np.minimum(h - 1, ymax))
      xmin_ = int(np.maximum(0, xmin))
      xmax_ = int(np.minimum(w - 1, xmax))
      img_crop = image[ymin_:ymax_ + 1, xmin_:xmax_ + 1]  # image
      mask_crop = mask[ymin_:ymax_ + 1, xmin_:xmax_ + 1]  # instance
      class_label_crop = (mask > 0)[ymin_:ymax_ + 1, xmin_:xmax_ + 1]  # foreground
      instance_list_tp.append({'id': id, 'global_position': [ymin, xmin], 'img_crop': img_crop, 'mask_crop': mask_crop,
                               'class_label_crop': class_label_crop, 'image_dims': [h, w]})
    save_pickle(os.path.join(crops_dir, os.path.basename(image_list[i])[:-4] + '.pkl'), instance_list_tp)


def create_seq_dicts(train_val_name, crops_dir='./crops', dicts_dir='./dicts'):
  """

    `create_seq_dicts` generates a dictionary `id_dictionary` where the keys of the
    dictionary are the present ids seen across time

    Parameters
    ----------
    train_val_name: str
          one of 'train' or 'val'
    dicts_dir: str
          for example, this could be 'dicts'

    Returns
    ----------

    """

  dicts_dir = os.path.join(dicts_dir, train_val_name)

  if not os.path.exists(dicts_dir):
    os.makedirs(os.path.dirname(dicts_dir))
    print("Created new directory : {}".format(dicts_dir))

  crops_list = sorted(glob(os.path.join(crops_dir, train_val_name, '*.pkl')))
  id_dictionary = {}

  for crop in tqdm(crops_list):  # each crop corresponds to one time point --> list of dictionaries
    instance_list = load_pickle(crop)
    for instance in instance_list:  # looping over all object ids -- > this gives one dictionary per object avatar
      id = instance['id']
      if (id not in id_dictionary.keys()):
        id_dictionary[id] = [instance]
      else:
        id_dictionary[id].append(instance)

  for key in id_dictionary.keys():
    save_pickle(os.path.join(dicts_dir, str(key).zfill(5) + '.pkl'), id_dictionary[key])


def pickle_data(data_dir, project_name, train_val_names=['train', 'val'], mode='2d'):
  """

  `pickle_data` essentially re-saves tracklets as pickled objects
  Parameters
  ----------
  train_val_names: list
        default is ['train', 'val']
  Returns
  ----------
  """
  input_dir = os.path.join(data_dir, project_name)
  for train_val_name in train_val_names:
    if mode == '2d':
      create_crops(input_dir, train_val_name + '/')
    else:
      create_crops_3d(input_dir, train_val_name + '/')
    if train_val_name == 'test':
      pass
    else:
      create_seq_dicts(train_val_name + '/')


def calculate_object_size(data_dir, project_name, train_val_name, background_id=0):
  """
  Calculate the mean object size from the available label masks

  Parameters
  -------

  data_dir: string
      Name of directory storing the data. For example, 'data'
  project_name: string
      Name of directory containing data specific to this project. For example, 'dsb-2018'
  train_val_name: string
      Name of directory containing 'train' and 'val' images and instance masks
  background_id: int
      Id which corresponds to the background.

  Returns
  -------
  (float, float, float, float, float, float, float)
  (minimum number of pixels in an object, mean number of pixels in an object, max number of pixels in an object,
  mean number of pixels along the y dimension,
  mean number of pixels along the x dimension,
  std number of pixels along the y dimension,
  std number of pixels along the x dimension,


  """

  instance_names = []
  size_list_x = []
  size_list_y = []
  size_list = []
  for name in train_val_name:
    instance_dir = os.path.join(data_dir, project_name, name, 'masks-reid')
    instance_names += sorted(glob(os.path.join(instance_dir, '*.tif')))

  n_images = len((instance_names))
  for i in tqdm(range(len(instance_names[:n_images])), position=0, leave=True):
    ma = tifffile.imread(instance_names[i])
    ids = np.unique(ma)
    ids = ids[ids != background_id]
    for id in ids:
      y, x = np.where(ma == id)
      size_list_x.append(np.max(x) - np.min(x))
      size_list_y.append(np.max(y) - np.min(y))
      size_list.append(len(x))

  print("Minimum object size of the `{}` dataset is equal to {}".format(project_name, np.min(size_list)))
  print("Mean object size of the `{}` dataset is equal to {}".format(project_name, np.mean(size_list)))
  print("Maximum object size of the `{}` dataset is equal to {}".format(project_name, np.max(size_list)))
  print("Average object size of the `{}` dataset along `x` is equal to {:.3f}".format(project_name,
                                                                                      np.mean(size_list_x)))
  print("Average object size of the `{}` dataset along `y` is equal to {:.3f}".format(project_name,
                                                                                      np.mean(size_list_y)))

  return np.min(size_list).astype(np.float), np.mean(size_list).astype(np.float), np.max(size_list).astype(
    np.float), np.mean(size_list_y).astype(np.float), np.mean(size_list_x).astype(np.float), np.std(size_list_y).astype(
    np.float), np.std(size_list_x).astype(np.float)


def calculate_object_size_3d(data_dir, project_name, train_val_name, background_id=0):
  """
  Calculate the mean object size from the available label masks

  Parameters
  -------

  data_dir: string
      Name of directory storing the data. For example, 'data'
  project_name: string
      Name of directory containing data specific to this project. For example, 'dsb-2018'
  train_val_name: string
      Name of directory containing 'train' and 'val' images and instance masks
  background_id: int
      Id which corresponds to the background.

  Returns
  -------
  (float, float, float, float, float, float, float, float, float)
  (minimum number of pixels in an object, mean number of pixels in an object, max number of pixels in an object,
  mean number of pixels along the z dimension,
  mean number of pixels along the y dimension,
  mean number of pixels along the x dimension,
  std number of pixels along the z dimension
  std number of pixels along the y dimension,
  std number of pixels along the x dimension,


  """

  instance_names = []
  size_list_x = []
  size_list_y = []
  size_list_z = []
  size_list = []

  for name in train_val_name:
    instance_dir = os.path.join(data_dir, project_name, name, 'masks-reid')
    instance_names += sorted(glob(os.path.join(instance_dir, '*.tif')))

  n_images = len((instance_names))
  for i in tqdm(range(len(instance_names[:n_images])), position=0, leave=True):
    ma = tifffile.imread(instance_names[i])
    ids = np.unique(ma)
    ids = ids[ids != background_id]
    for id in ids:
      z, y, x = np.where(ma == id)
      size_list_x.append(np.max(x) - np.min(x))
      size_list_y.append(np.max(y) - np.min(y))
      size_list_z.append(np.max(z) - np.min(z))
      size_list.append(len(x))

  print("Minimum object size of the `{}` dataset is equal to {}".format(project_name, np.min(size_list)))
  print("Mean object size of the `{}` dataset is equal to {}".format(project_name, np.mean(size_list)))
  print("Maximum object size of the `{}` dataset is equal to {}".format(project_name, np.max(size_list)))
  print("Average object size of the `{}` dataset along `x` is equal to {:.3f}".format(project_name,
                                                                                      np.mean(size_list_x)))
  print("Average object size of the `{}` dataset along `y` is equal to {:.3f}".format(project_name,
                                                                                      np.mean(size_list_y)))
  print("Average object size of the `{}` dataset along `z` is equal to {:.3f}".format(project_name,
                                                                                      np.mean(size_list_z)))
  return np.min(size_list).astype(np.float), np.mean(size_list).astype(np.float), np.max(size_list).astype(
    np.float), np.mean(size_list_z).astype(np.float), np.mean(size_list_y).astype(np.float), np.mean(
    size_list_x).astype(np.float), \
         np.std(size_list_z).astype(np.float), np.std(size_list_y).astype(np.float), np.std(size_list_x).astype(
    np.float)


def calculate_avg_num_tracklets(data_dir, project_name, train_val_name, background_id=0):
  """
  Calculate the mean object size from the available label masks

  Parameters
  -------

  data_dir: string
      Name of directory storing the data. For example, 'data'
  project_name: string
      Name of directory containing data specific to this project. For example, 'dsb-2018'
  train_val_name: string
      Name of directory containing 'train' and 'val' images and instance masks
  background_id: int
      Id which corresponds to the background.

  Returns
  -------
   (float, float, float)
   min number of ids, mean number of ids, max number of ids
  """
  instance_names = []
  num_ids = []
  for name in train_val_name:
    instance_dir = os.path.join(data_dir, project_name, name, 'masks-reid')
    instance_names += sorted(glob(os.path.join(instance_dir, '*.tif')))

  n_images = len((instance_names))
  for i in tqdm(range(len(instance_names[:n_images])), position=0, leave=True):
    ma = tifffile.imread(instance_names[i])
    ids = np.unique(ma)
    ids = ids[ids != background_id]
    num_ids.append(len(ids))
  print("Minimum number of tracklets in the `{}` dataset is equal to {}".format(project_name, np.min(num_ids)))
  print("Minimum number of tracklets in the `{}` dataset is equal to {}".format(project_name, np.mean(num_ids)))
  print("Maximum number of tracklets in the `{}` dataset is equal to {}".format(project_name, np.max(num_ids)))
  return np.min(num_ids).astype(np.float), np.mean(num_ids).astype(np.float), np.max(num_ids).astype(np.float)


def get_tracklet_length(data_dir, project_name, train_val_name, background_id=0):
  """
  Calculate the tracklet length from the available label masks

  Parameters
  -------

  data_dir: string
      Name of directory storing the data. For example, 'data'
  project_name: string
      Name of directory containing data specific to this project. For example, 'dsb-2018'
  train_val_name: string
      Name of directory containing 'train' and 'val' images and instance masks
  background_id: int
      Id which corresponds to the background.

  Returns
  -------
   (float, float, float, float)
   min length of tracklet, mean length of tracklet, max length of tracklet, std. dev of tracklet
  """
  instance_names = []
  for name in train_val_name:
    instance_dir = os.path.join(data_dir, project_name, name, 'masks-reid')
    instance_names += sorted(glob(os.path.join(instance_dir, '*.tif')))

  n_images = len(instance_names)

  tracklet_dict = {}
  for i in tqdm(range(len(instance_names[:n_images])), position=0, leave=True):
    ma = tifffile.imread(instance_names[i])
    ids = np.unique(ma)
    ids = ids[ids != background_id]
    for id in ids:
      if id in tracklet_dict.keys():
        tracklet_dict[id] += 1
      else:
        tracklet_dict[id] = 0

  length = []

  for item in tracklet_dict.items():
    length.append(item[1])
  print("Minimum length of tracklet in the `{}` dataset is equal to {}".format(project_name, np.min(length)))
  print("Mean number of tracklets in the `{}` dataset is equal to {}".format(project_name, np.mean(length)))
  print("Maximum number of tracklets in the `{}` dataset is equal to {}".format(project_name, np.max(length)))
  print("Std. dev. of tracklets in the `{}` dataset is equal to {}".format(project_name, np.std(length)))
  return np.min(length).astype(np.float), np.mean(length).astype(np.float), np.max(length).astype(np.float), np.std(
    length).astype(np.float)


def get_data_properties(data_dir, project_name, train_val_name, mode='2d', background_id=0):
  """

  Parameters
  -------

  data_dir: string
          Path to directory containing all data
  project_name: string
          Path to directory containing project-specific images and instances
  train_val_name: string
          One of 'train' or 'val'
  mode: string
          One of '2d' or '3d'
  background_id: int
          Label id corresponding to the background

  Returns
  -------
  data_properties_dir: dictionary
          keys include `foreground_weight`, `min_object_size`, `project_name`, `avg_background_intensity` etc

  """
  data_properties_dir = {}
  if mode == '2d':
    data_properties_dir['min_object_size'], data_properties_dir['mean_object_size'], data_properties_dir[
      'max_object_size'], data_properties_dir['avg_object_size_y'], data_properties_dir[
      'avg_object_size_x'], data_properties_dir['std_object_size_y'], data_properties_dir[
      'std_object_size_x'] = calculate_object_size(data_dir, project_name, train_val_name=train_val_name)
  else:
    data_properties_dir['min_object_size'], data_properties_dir['mean_object_size'], data_properties_dir[
      'max_object_size'], data_properties_dir['avg_object_size_z'], data_properties_dir['avg_object_size_y'], \
    data_properties_dir['avg_object_size_x'], data_properties_dir['std_object_size_z'], data_properties_dir[
      'std_object_size_y'], \
    data_properties_dir['std_object_size_x'] = calculate_object_size_3d(data_dir, project_name,
                                                                        train_val_name=train_val_name)

  data_properties_dir['min_num_tracklets'], data_properties_dir['mean_num_tracklets'], data_properties_dir[
    'max_num_tracklets'] = calculate_avg_num_tracklets(data_dir, project_name, train_val_name)
  data_properties_dir['min_length_tracklet'], data_properties_dir['mean_length_tracklet'], \
  data_properties_dir['max_length_tracklet'], data_properties_dir['std_length_tracklet'] = \
    get_tracklet_length(data_dir, project_name, train_val_name)
  data_properties_dir['project_name'] = project_name
  return data_properties_dir
