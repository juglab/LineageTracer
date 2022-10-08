import os
import shutil
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from LineageTracer.criterions import get_loss
from LineageTracer.datasets import get_dataset
from LineageTracer.models import get_model
from LineageTracer.utils.utils import AverageMeter, Logger

torch.backends.cudnn.benchmark = True
import numpy as np


def train():
  # define meters
  loss_meter = AverageMeter()

  # put model into training mode
  model.train()

  for param_group in optimizer.param_groups:
    print('learning rate: {}'.format(param_group['lr']))

  for i, sample in enumerate(tqdm(train_dataset_it)):
    point_features = sample['point_features']
    # example: (1, 72, 100, 131)
    # first dimension is batch-size,
    # second dimension is 3*num_tracklets,
    # third dimension is num_sampled_points

    normalized_global = sample['normalized_global']
    # example: (1, 72, 4) or (1, 72, 6)
    labels = sample['labels']  # (1, 72)
    embeddings = model(point_features, normalized_global)  # (72, 32)
    loss = criterion(embeddings, labels)
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_meter.update(loss.item())

  return loss_meter.avg


def val():
  # define meters
  loss_meter = AverageMeter()

  # put model into eval mode
  model.eval()

  with torch.no_grad():
    for i, sample in enumerate(tqdm(val_dataset_it)):
      point_features = sample['point_features']  # (1, 72, 100, 131) --> first dimension is batch-size
      normalized_global = sample['normalized_global']  # (1, 72, 4)
      labels = sample['labels']  # (1, 72)
      embeddings = model(point_features, normalized_global)  # (72, 32)
      loss = criterion(embeddings, labels)
      loss = loss.mean()
      loss_meter.update(loss.item())

  return loss_meter.avg


def save_checkpoint(state, is_best, epoch, save_dir, save_checkpoint_frequency, name='checkpoint.pth'):
  """Saves checkpoint.
    Parameters
    ----------
    state : dictionary
        The state of the model weights
    is_best : bool
        In case the validation IoU is higher at the end of a certain epoch than previously recorded, `is_best` is set equal to True
    epoch: int
        The current epoch
    save_checkpoint_frequency: int
        The model weights are saved every `save_checkpoint_frequency` epochs
    name: str, optional
        The model weights are saved under the name `name`
    Returns
    -------
    """
  print('=> saving checkpoint')
  file_name = os.path.join(save_dir, name)
  torch.save(state, file_name)
  if (save_checkpoint_frequency is not None):
    if (epoch % int(save_checkpoint_frequency) == 0):
      file_name2 = os.path.join(save_dir, str(epoch) + "_" + name)
      torch.save(state, file_name2)
  if is_best:
    shutil.copyfile(file_name, os.path.join(save_dir, 'best_model.pth'))


def begin_training(train_dataset_dict, val_dataset_dict, model_dict, loss_dict, configs):
  """Entry function for beginning the model training procedure.

              Parameters
              ----------
              train_dataset_dict : dictionary
                  Dictionary containing training data loader-specific parameters (for e.g. train_batch_size etc)
              val_dataset_dict : dictionary
                  Dictionary containing validation data loader-specific parameters (for e.g. val_batch_size etc)
              model_dict: dictionary
                  Dictionary containing model specific parameters (for e.g. number of outputs)
              loss_dict: dictionary
                  Dictionary containing loss specific parameters (for e.g. convex weights of different loss terms - w_iou, w_var etc)
              configs: dictionary
                  Dictionary containing general training parameters (for e.g. num_epochs, learning_rate etc)

              Returns
              -------
              """

  if configs['save']:
    if not os.path.exists(configs['save_dir']):
      os.makedirs(configs['save_dir'])

  # set device
  device = torch.device("cuda:0" if configs['cuda'] else "cpu")

  # define global variables
  global train_dataset_it, val_dataset_it, model, criterion, optimizer

  # train dataloader
  train_dataset = get_dataset(train_dataset_dict['type'], train_dataset_dict['kwargs'])
  train_dataset_it = torch.utils.data.DataLoader(train_dataset, batch_size=train_dataset_dict['batch_size'],
                                                 shuffle=True, drop_last=True,
                                                 num_workers=train_dataset_dict['workers'],
                                                 pin_memory=True if configs['cuda'] else False)

  # val dataloader
  val_dataset = get_dataset(val_dataset_dict['type'], val_dataset_dict['kwargs'])
  val_dataset_it = torch.utils.data.DataLoader(val_dataset, batch_size=val_dataset_dict['batch_size'], shuffle=True,
                                               drop_last=False, num_workers=val_dataset_dict['workers'],
                                               pin_memory=True if configs['cuda'] else False)

  # set model
  model = get_model(model_dict['name'], model_dict['kwargs'])
  model.init_output()
  model = torch.nn.DataParallel(model).to(device)

  # set criterion
  criterion = get_loss(loss_opts=loss_dict['margin'])
  criterion = torch.nn.DataParallel(criterion).to(device)

  # set optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=configs['train_lr'], weight_decay=1e-4)

  def lambda_(epoch):
    return pow((1 - ((epoch) / 200)), 0.9)

  # Logger
  logger = Logger(('train', 'val'), 'loss')

  # resume
  start_epoch = 0
  lowest_loss = 1e2

  if configs['resume_path'] is not None and os.path.exists(configs['resume_path']):
    print('Resuming model from {}'.format(configs['resume_path']))
    state = torch.load(configs['resume_path'])
    start_epoch = state['epoch'] + 1
    lowest_loss = state['lowest_loss']
    model.load_state_dict(state['model_state_dict'], strict=True)
    optimizer.load_state_dict(state['optim_state_dict'])
    logger.data = state['logger_data']

  for epoch in range(start_epoch, configs['n_epochs']):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_, last_epoch=epoch - 1)
    print('Starting epoch {}'.format(epoch))
    train_loss = train()
    val_loss = val()
    scheduler.step()
    print('===> train loss: {:.3f}'.format(train_loss))
    print('===> val loss: {:.3f}'.format(val_loss))

    logger.add('train', train_loss)
    logger.add('val', val_loss)
    logger.plot(save=configs['save'], save_dir=configs['save_dir'])

    is_best = val_loss < lowest_loss
    lowest_loss = min(val_loss, lowest_loss)

    if configs['save']:
      state = {
        'epoch': epoch,
        'lowest_loss': lowest_loss,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'logger_data': logger.data,
      }
    save_checkpoint(state, is_best, epoch, save_dir=configs['save_dir'],
                    save_checkpoint_frequency=configs['save_checkpoint_frequency'])
