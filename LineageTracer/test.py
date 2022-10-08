import numpy as np
import os
import torch
from glob import glob
from tqdm import tqdm

from LineageTracer.datasets import get_dataset
from LineageTracer.models import get_model

torch.backends.cudnn.benchmark = True
import tifffile
import pandas as pd
import json
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import shutil
import subprocess
import shlex


def test(dataset_it, model, save_dir, embedding_path):
  """
    Function called by `output_embeddings`
    Loops over each crop from the test image one-by-one and saves the output embeddings to `save_dir`

    Parameters
    ----------
    dataset_it: Data Loader
        Test data loader

    model: PyTorch Model
        Trained model weights
    save_dir: str
        Path to where the intermediate and final label masks are saved
    Returns
    ----------

    """
  model.eval()

  with torch.no_grad():
    for i, sample in enumerate(tqdm(dataset_it)):
      if len(sample['point_features']) != 0:
        point_features = sample[
          'point_features']  # (1, nobjects/tp=72, npoints=100, ndims=131) --> first dimension is batch-size
        normalized_global = sample['normalized_global']  # (1, 72, 4)
        embeddings = model(point_features, normalized_global)  # (72, 32)

        object_ids = sample['object_id'][0]  # (50 objects)
        time_points = sample['time_point'][0]  # (50 objects)
        embeddings_cpu = embeddings.cpu().detach().numpy()

        for tp, embedding in enumerate(embeddings_cpu):
          # embedding has shape (32,)
          np.save(
            embedding_path + '/' + str(time_points[tp].item()).zfill(4) + '_' + str(object_ids[tp].item()).zfill(
              4),
            embedding)


def output_embeddings(test_configs):
  """
    In this step, we input crops extracted from the test images one by one into the model and generate corresponding embeddings.

    Parameters
    ----------
    test_configs: dictionary

    Returns
    ----------

    """
  # set device
  device = torch.device("cuda:0" if test_configs['cuda'] else "cpu")

  # dataloader
  dataset = get_dataset(test_configs['dataset']['type'], test_configs['dataset']['kwargs'])
  dataset_it = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False,
                                           num_workers=test_configs['num_workers'],
                                           pin_memory=True if test_configs['cuda'] else False)

  # load model
  model = get_model(test_configs['model']['name'], test_configs['model']['kwargs'])
  model = torch.nn.DataParallel(model).to(device)

  # load snapshot
  if os.path.exists(test_configs['checkpoint_path']):
    state = torch.load(test_configs['checkpoint_path'])
    model.load_state_dict(state['model_state_dict'], strict=True)
  else:
    assert False, 'checkpoint_path {} does not exist!'.format(test_configs['checkpoint_path'])

  save_dir = test_configs['save_dir']
  save_images = test_configs['save_images']
  project_name = test_configs['project_name']
  if save_images:
    if not os.path.exists(os.path.join(save_dir, project_name + '_embeddings/')):
      os.makedirs(os.path.join(save_dir, project_name + '_embeddings/'))
      print("Created new directory {}".format(os.path.join(save_dir, project_name + '_embeddings/')))

  test(dataset_it, model, save_dir, embedding_path=os.path.join(save_dir, project_name + '_embeddings/'))


def compute_tracklets(test_configs):
  """
  In this step, we identify small tracklets where the learning criteria, going forward in time, is satisfied
  i.e. distance of an anchor at time point $t$ to the distance of a positive sample at time point $t+1$ + margin distance is less
  than distance of an anchor at time point $t$ to the distance of a negative sample at time ppoint $t+1$.
  Obviously, this would not be able to correct for **over segmentations**, identify **under-segmentations** or handle divisions.
  These things are corrected for later.

  Parameters
  ----------
  test_configs: dictionary

  Returns
  ----------

  """
  mask_filenames = sorted(glob(test_configs['data_dir'] + '/test/masks/*.tif'))
  gt_track_filenames = sorted(glob(test_configs['data_dir'] + '/test/tracking-annotations/*.tif'))
  gt_track_filenames.append(test_configs['data_dir'] + '/test/tracking-annotations/man_track.txt')
  image_filenames = sorted(glob(test_configs['data_dir'] + '/test/images/*.tif'))
  project_name = test_configs['project_name']
  save_dir = test_configs['save_dir']
  save_images = test_configs['save_images']
  save_results = test_configs['save_results']
  embeddings_dir = os.path.join(test_configs['save_dir'], project_name + '_embeddings')
  margin = test_configs['loss_dict']['margin']

  if save_images:
    if not os.path.exists(os.path.join(save_dir, project_name)):  # images
      os.makedirs(os.path.join(save_dir, project_name))
      print("Created new directory {}".format(os.path.join(save_dir, project_name)))
    if not os.path.exists(os.path.join(save_dir, project_name + '_tracklets/')):  # result directory
      os.makedirs(os.path.join(save_dir, project_name + '_tracklets/'))
      print("Created new directory {}".format(os.path.join(save_dir, project_name + '_tracklets/')))
    if not os.path.exists(os.path.join(save_dir, project_name + '_GT/TRA/')):  # ground-truth directory
      os.makedirs(os.path.join(save_dir, project_name + '_GT/TRA/'))
      print("Created new directory {}".format(os.path.join(save_dir, project_name + '_GT/TRA/')))
    if not os.path.exists(os.path.join(save_dir, project_name + '_dist_matrices/')):  # distance matrices
      os.makedirs(os.path.join(save_dir, project_name + '_dist_matrices/'))
      print("Created new directory {}".format(os.path.join(save_dir, project_name + '_dist_matrices/')))

  for filename in gt_track_filenames:
    shutil.copyfile(filename, os.path.join(save_dir, project_name + '_GT/TRA/' + os.path.basename(filename)))

  for filename in image_filenames:
    shutil.copyfile(filename, os.path.join(save_dir, project_name + '/' + os.path.basename(filename)))
  tracklet_dir = os.path.join(save_dir, project_name + '_tracklets/')
  json_dir = os.path.join(save_dir, project_name + '_tracklets/')
  dist_matrices_dir = os.path.join(save_dir, project_name + '_dist_matrices/')

  old_id_new_id_map = {}  # map from ids in segmentation masks to relabeled id
  new_id_ts_te_map = {}  # keeps track of how long an id propagates

  id_counter = 1
  dist_match_list = []
  for t in tqdm(range(len(mask_filenames) - 1)):

    old_id_old_id_map = {}  # distance between learnt embeddings
    im_t = tifffile.imread(mask_filenames[t])
    im_tp1 = tifffile.imread(mask_filenames[t + 1])

    ids_t = np.unique(im_t)
    ids_t = ids_t[ids_t != 0]

    ids_tp1 = np.unique(im_tp1)
    ids_tp1 = ids_tp1[ids_tp1 != 0]

    if len(ids_t) == 0:  # i.e. no ids detected in im_t # TODO (should be handled as well)
      pass

    elif len(ids_tp1) == 0:  # i.e. no ids detected in im_tp1 # TODO (should be handled as well)
      pass

    else:
      dist_matrix = np.zeros((len(ids_t), len(ids_tp1)), dtype=np.float32)
      processed_start = ids_t.tolist()
      processed_end = ids_tp1.tolist()
      for i in range(dist_matrix.shape[0]):
        v_i = np.load(embeddings_dir + '/' + str(t).zfill(4) + '_' + str(ids_t[i]).zfill(4) + '.npy')
        for j in range(dist_matrix.shape[1]):
          v_j = np.load(
            embeddings_dir + '/' + str(t + 1).zfill(4) + '_' + str(ids_tp1[j]).zfill(4) + '.npy')
          dist_matrix[i, j] = np.linalg.norm([v_i - v_j])
          old_id_old_id_map[str(ids_t[i]) + '_' + str(ids_tp1[j])] = str(dist_matrix[i, j])

      with open(dist_matrices_dir + '/' + str(t).zfill(3) + '.json', 'w') as fp:
        json.dump(old_id_old_id_map, fp)

      # just find index minima for all input candidates i
      indices_best = np.argmin(dist_matrix, 1)

      # first we mask out (effectively set to zero in the distance matrix) candidates for which the learning criteria is not satisfied
      mask = np.zeros(indices_best.shape, dtype=np.bool)

      for i in range(dist_matrix.shape[0]):
        indices = np.argsort(dist_matrix[i, :])
        if (dist_matrix[i, indices[0]] + margin < dist_matrix[i, indices[1]]):
          mask[i] = True

      # first we identify the ones that meet the criterion

      for i in range(dist_matrix.shape[0]):
        indices = np.argsort(dist_matrix[i, :])
        if (dist_matrix[i, indices[0]] + margin < dist_matrix[i, indices[1]]) and (
          np.count_nonzero(indices_best[mask] == indices[0]) == 1):
          if ids_t[i] in processed_start:
            processed_start.remove(ids_t[i])
          if ids_tp1[indices[0]] in processed_end:
            processed_end.remove(ids_tp1[indices[0]])

          dist_match_list.append(dist_matrix[i, indices[0]])
          # i.e. the distance of anchor to positive + margin distance is less than distance of anchor to a negative sample
          # the `and` ensures that multiple objects at time `t` do not think that the same object at `tp1` is their positive match
          if str(t) + '_' + str(ids_t[i]) in old_id_new_id_map.keys():
            old_id_new_id_map[str(t + 1) + '_' + str(ids_tp1[indices[0]])] = old_id_new_id_map[
              str(t) + '_' + str(ids_t[i])]
            new_id_ts_te_map[old_id_new_id_map[str(t) + '_' + str(ids_t[i])]][1] = t + 1
          else:
            old_id_new_id_map[str(t) + '_' + str(ids_t[i])] = id_counter
            old_id_new_id_map[str(t + 1) + '_' + str(ids_tp1[indices[0]])] = id_counter
            new_id_ts_te_map[id_counter] = np.array([t, t + 1])
            id_counter += 1

      # then we identify the ones where there is an iou overlap


      processed_start_copy = processed_start.copy()
      processed_end_copy = processed_end.copy()
      for i in processed_start_copy:
        for j in processed_end_copy:
          intersection = ((im_t == i) & (im_tp1 == j)).sum()
          if intersection > 0:
            if i in processed_start:
              processed_start.remove(i)
            if j in processed_end:
              processed_end.remove(j)
            if str(t) + '_' + str(i) in old_id_new_id_map.keys():
              # also check if tp1, j is already assigned by chance?
              if str(t + 1) + '_' + str(j) in old_id_new_id_map.keys(): # edge case
                pass
              else:
                old_id_new_id_map[str(t + 1) + '_' + str(j)] = old_id_new_id_map[
                str(t) + '_' + str(i)]
                new_id_ts_te_map[old_id_new_id_map[str(t) + '_' + str(i)]][1] = t + 1
            else:
              if str(t + 1) + '_' + str(j) in old_id_new_id_map.keys(): # edge case
                old_id_new_id_map[str(t) + '_' + str(i)] = id_counter
                new_id_ts_te_map[id_counter] = np.array([t, t])
                id_counter+=1
              else:
                old_id_new_id_map[str(t) + '_' + str(i)] = id_counter
                old_id_new_id_map[str(t + 1) + '_' + str(j)] = id_counter
                new_id_ts_te_map[id_counter] = np.array([t, t + 1])
                id_counter += 1


      # else we just leave the confusing lot and hope they are fixed in the next ILP step!
      processed_start_copy = processed_start.copy()

      for i in processed_start_copy:
        if str(t) + '_' + str(i) in old_id_new_id_map.keys():
          pass
        else:
          old_id_new_id_map[str(t) + '_' + str(i)] = id_counter
          new_id_ts_te_map[id_counter] = np.array([t, t])
          id_counter += 1

  # process last time point additionally
  t = len(mask_filenames) - 1
  im_last = tifffile.imread(mask_filenames[t])
  ids_t = np.unique(im_last)
  ids_t = ids_t[ids_t != 0]

  for id in ids_t:
    if str(t) + '_' + str(id) in old_id_new_id_map.keys():
      pass
    else:
      old_id_new_id_map[str(t) + '_' + str(id)] = id_counter
      new_id_ts_te_map[id_counter] = np.array([t, t])
      id_counter += 1

  with open(json_dir + '/' + 'old_id_new_id_map_create_tracklets.json', 'w') as fp:
    json.dump(old_id_new_id_map, fp)

  # Build reverse start ids
  reverse_tracklet_id_instance_id_start = {}  # map of new id to where the id was seen for the first time

  for key, value in old_id_new_id_map.items():
    t, id = key.split('_')
    t, id = int(t), int(id)
    if value in reverse_tracklet_id_instance_id_start.keys():
      t_temp, id_temp = reverse_tracklet_id_instance_id_start[value].split('_')
      t_temp, id_temp = int(t_temp), int(id_temp)
      if t_temp > t:
        reverse_tracklet_id_instance_id_start[value] = key
    else:
      reverse_tracklet_id_instance_id_start[value] = key

  with open(json_dir + '/' + '/reverse_tracklet_id_instance_id_start.json', 'w') as fp:
    json.dump(reverse_tracklet_id_instance_id_start, fp)

  # Build reverse end ids
  reverse_tracklet_id_instance_id_end = {}  # map of new id to where the id was seen for the last time
  for key, value in old_id_new_id_map.items():
    t, id = key.split('_')
    t, id = int(t), int(id)
    if value in reverse_tracklet_id_instance_id_end.keys():
      t_temp, id_temp = reverse_tracklet_id_instance_id_end[value].split('_')
      t_temp, id_temp = int(t_temp), int(id_temp)
      if t_temp < t:
        reverse_tracklet_id_instance_id_end[value] = key
    else:
      reverse_tracklet_id_instance_id_end[value] = key

  with open(json_dir + '/' + '/reverse_tracklet_id_instance_id_end.json', 'w') as fp:
    json.dump(reverse_tracklet_id_instance_id_end, fp)

  # Create `res_track.txt`
  res_list = []
  for key in new_id_ts_te_map.keys():
    res_list.append(
      np.array([key, new_id_ts_te_map[key][0], new_id_ts_te_map[key][1], 0]))  # this step does not handle divisions yet

  df = pd.DataFrame(res_list)
  df.to_csv(tracklet_dir + '/' + 'res_track.txt', index=False, header=False, sep=' ')

  for t in range(len(mask_filenames)):
    ma = tifffile.imread(mask_filenames[t])
    ma_empty = np.zeros_like(ma)
    tifffile.imsave(tracklet_dir + '/' + 'mask' + str(t).zfill(3) + '.tif', ma_empty)

  # Put the correct tracklet id
  for key in tqdm(old_id_new_id_map.keys()):
    t, id = key.split('_')
    t, id = int(t), int(id)
    ma = tifffile.imread(mask_filenames[t])
    ma_tracklet = tifffile.imread(tracklet_dir + '/' + 'mask' + str(t).zfill(3) + '.tif')
    ma_tracklet[ma == id] = old_id_new_id_map[key]
    tifffile.imsave(tracklet_dir + '/' + 'mask' + str(t).zfill(3) + '.tif', ma_tracklet)




  if save_results:
    os.rename(tracklet_dir, os.path.join(save_dir, project_name + '_RES/'))
    subprocess.check_call(["../../../LineageTracer/utils/TRAMeasure", save_dir, project_name, '3'], shell=False)
    os.rename(os.path.join(save_dir, project_name + '_RES/'), tracklet_dir)
  return np.mean(dist_match_list), np.std(dist_match_list)


def stitch_tracklets(test_configs, max_dist_embedding=0.75, cost_app=1.0, cost_disapp=1.0, dT=3, scale_factor=100):
  """
    In this step, we stitch tracklets produced by `compute_tracklets` function to form a lineage tree.
    Firstly, we create a directed graph.
    Each tracklet is considered as one node.
    Edges are built between nodes(tracklets) which are contiguous in time
    (contiguous implies that say if one tracklet finishes and another tracklet starts "soon" after, then an edge is drawn between these nodes / tracklets)

    Regarding constraints, we make sure that:
    (1)  X_nodes_app[node] + previous indicators == 1
    (this ensures that there is only one, at the most, incoming edge ... )
    (2)  2 * X_nodes_disappearance[node] + next_indicators <= 2
    (this ensures that, at the most, two daughter cells.
    But this could also mean that both terms on the LHS are zero ... Hence, the last constraint!)
    (3) 2 * X_nodes_disappearance[node] + next_indicators >= 1


    Parameters
    ----------
    test_configs: dictionary

    max_dist_embedding: float
        While building a graph, we only draw edges between tracklets, where we see the distance between the embeddings less than `max_dist_embedding`
    cost_app: float
        default = 1.0
    cost_disapp: float
        default = 1.0
    dT : int
        Edges can be built even if there is a temporal gap between start and end of tracklets


    Returns
    ----------

    """
  mask_filenames = sorted(glob(test_configs['data_dir'] + '/test/masks/*.tif'))
  save_dir = test_configs['save_dir']
  save_images = test_configs['save_images']
  save_results = test_configs['save_results']
  project_name = test_configs['project_name']
  if save_images:
    if not os.path.exists(os.path.join(save_dir, project_name + '_ILP/')):
      os.makedirs(os.path.join(save_dir, project_name + '_ILP/'))
      print("Created new directory {}".format(os.path.join(save_dir, project_name + '_ILP/')))

  ilp_dirname = os.path.join(save_dir, project_name + '_ILP/')
  embeddings_dirname = os.path.join(save_dir, project_name + '_embeddings/')
  tracklets_dirname = os.path.join(save_dir, project_name + '_tracklets/')

  # initialize directed graph
  G = nx.DiGraph()

  # Add nodes (tracklets) to directed graph

  # assume that a tra like text file has been generated from the previous step with the parent id set to 0 always (erroneously)
  df = pd.read_csv(tracklets_dirname + '/res_track.txt', delimiter=' ', header=None)
  df_ = df.to_numpy()

  for row in tqdm(df_):
    G.add_node(str(row[0]), cost_appearance=cost_app, cost_disappearance=cost_disapp)

  # Next, add edges to directed graph
  import json
  with open(os.path.join(tracklets_dirname + 'reverse_tracklet_id_instance_id_end.json')) as json_file:
    reverse_data_end = json.load(json_file)

  with open(os.path.join(tracklets_dirname + 'reverse_tracklet_id_instance_id_start.json')) as json_file:
    reverse_data_start = json.load(json_file)

  # Adding edges to directed graph
  for i in range(df_.shape[0]):
    for j in range(df_.shape[0]):
      if 0 < (df_[j, 1] - df_[i, 2]) and (df_[j, 1] - df_[i, 2]) < dT and df_[i, 0] != df_[j, 0]:
        # we are checking if a certain tracklet starts after another tracklet finishes
        # and also if the time gap between the start of that tracklet and finish of another tracklet is less than $dT$
        # and that we are looking at two distinct tracklets!
        time_id_start_original = reverse_data_start[str(df_[j, 0])]
        time_start, id_start = time_id_start_original.split('_')
        time_start, id_start = int(time_start), int(id_start)
        time_id_end_original = reverse_data_end[str(df_[i, 0])]
        time_end, id_end = time_id_end_original.split('_')
        time_end, id_end = int(time_end), int(id_end)
        if time_end != df_[i, 2]:
          print("error", time_end, df_[i, 2])
        if time_start != df_[j, 1]:
          print("error", time_start, df_[j, 1])
        v_i = np.load(
          embeddings_dirname + '/' + str(df_[i, 2]).zfill(4) + '_' + str(id_end).zfill(4) + '.npy')
        v_j = np.load(
          embeddings_dirname + '/' + str(df_[j, 1]).zfill(4) + '_' + str(id_start).zfill(4) + '.npy')
        if np.linalg.norm(v_i - v_j) < max_dist_embedding:
          G.add_edge(str(df_[i, 0]), str(df_[j, 0]), cost_edge=np.linalg.norm(v_i - v_j) / scale_factor)

  # initialize gurobi model
  model = gp.Model()

  # Initialize indicator variables
  # We have 0/1 indicator variables only on the edges
  # (since most of the DL-generated segmentations are often good, we don't feel a need to throw away any segmentation
  # hence, we don't solve for the X_nodes)

  X_nodes = {}
  X_nodes_appearance = {}
  X_nodes_disappearance = {}
  X_edges = {}

  # add indicator variables on nodes to model
  for row in tqdm(df_):
    # boundary condition
    if row[1] == 0:  # if start timepoint is 0
      X_nodes_appearance[str(row[0])] = 1.0
    else:
      X_nodes_appearance[str(row[0])] = model.addVar(vtype=GRB.BINARY, name="X_app_%d" % (row[0]))
    # boundary condition
    if row[2] == len(mask_filenames) - 1:  # if final timepoint is last
      X_nodes_disappearance[str(row[0])] = 1.0
    else:
      X_nodes_disappearance[str(row[0])] = model.addVar(vtype=GRB.BINARY, name="X_disapp_%d" % (row[0]))

  # add indicator variables on edges to model
  for i in range(df_.shape[0]):
    for j in range(df_.shape[0]):
      if 0 < (df_[j, 1] - df_[i, 2]) and (df_[j, 1] - df_[i, 2]) < dT and df_[i, 0] != df_[j, 0]:

        time_id_start_original = reverse_data_start[str(df_[j, 0])]
        time_start, id_start = time_id_start_original.split('_')
        time_start, id_start = int(time_start), int(id_start)
        time_id_end_original = reverse_data_end[str(df_[i, 0])]
        time_end, id_end = time_id_end_original.split('_')
        time_end, id_end = int(time_end), int(id_end)

        v_i = np.load(
          embeddings_dirname + '/' + str(df_[i, 2]).zfill(4) + '_' + str(id_end).zfill(4) + '.npy')
        v_j = np.load(
          embeddings_dirname + '/' + str(df_[j, 1]).zfill(4) + '_' + str(id_start).zfill(4) + '.npy')

        if np.linalg.norm(v_i - v_j) < max_dist_embedding:
          X_edges[str(df_[i, 0]) + '_' + str(df_[j, 0])] = model.addVar(vtype=GRB.BINARY,
                                                                        name="X_edge_%d_%d" % (df_[i, 0], df_[j, 0]))

  model.update()
  model.modelSense = GRB.MINIMIZE

  # add constraints
  constraints = []

  # constraint on incoming edges!

  for node, cost_dic in G.nodes(data=True):
    previous_indicators = 0
    # sum up incoming edges
    for edge in G.in_edges(node):  # tuple of (start_node, end_node)
      id2 = edge[0]
      previous_indicators += X_edges[id2 + '_' + node]
    constraints.append(model.addConstr(1.0 == X_nodes_appearance[node] + previous_indicators,
                                       'constraint_continuation_previous_%s' % (node)))

    next_indicators = 0
    # sum up outgoing edges
    for edge in G.out_edges(node):  # tuple of (start_node, end_node)
      id2 = edge[1]  # incoming nodes will be index 0
      next_indicators += X_edges[node + '_' + id2]
    constraints.append(model.addConstr(2.0 >= 2 * X_nodes_disappearance[node] + next_indicators,
                                       'constraint_continuation_next_1_%s' % (
                                         node)))
    constraints.append(model.addConstr(1.0 <= 2 * X_nodes_disappearance[node] + next_indicators,
                                       'constraint_continuation_next_2_%s' % (node)))

  # find final result
  objective = 0
  for node, cost_dic in G.nodes(data=True):
    objective += cost_dic['cost_appearance'] * X_nodes_appearance[node] + cost_dic['cost_disappearance'] * \
                 X_nodes_disappearance[node]

  for node_u, node_v, cost_dic in G.edges(data=True):
    objective += cost_dic['cost_edge'] * X_edges[node_u + '_' + node_v]

  model.setObjective(objective)
  model.update()
  model.optimize()

  print('Total runtime is {}'.format(model.Runtime))

  print("Saving result to text file ...")

  results_file = []
  for v in model.getVars():
    results_file.append([v.VarName, v.X])

  with open(ilp_dirname + '/gurobi_vars.txt', 'w') as f:
    for line in results_file:
      f.write(f"{line}\n")

  print("Create res_track.txt")

  old_new_id_map = {}  # keys are all strings
  new_id_ts_te_map = {}  # keys are all integers
  key_count = 1

  # first only consider tracklets which should be kept
  for row in tqdm(df_):

    # check if id exists in new dictionary
    if str(row[0]) in old_new_id_map.keys():
      pass
    else:
      old_new_id_map[str(row[0])] = key_count
      new_id_ts_te_map[key_count] = np.array([row[1], row[2], 0])
      key_count += 1

  for node, cost_dic in G.nodes(data=True):
    id_daughters = []
    for edge in G.out_edges(node):  # tuple of (start_node, end_node)
      id2 = edge[1]  # start nodes will be index 0
      if model.getVarByName('X_edge_' + node + '_' + str(id2)).X == 1:
        id_daughters.append(id2)

    if len(id_daughters) == 1:  # actually a linking event!
      new_id_ts_te_map[old_new_id_map[node]][1] = new_id_ts_te_map[old_new_id_map[id_daughters[0]]][
        1]  # move the last time point further # TODO what if there are gaps?
      new_id_ts_te_map.pop(old_new_id_map[id_daughters[0]])  # the daughter shouldn't exist as an independent entity
      old_new_id_map[id_daughters[0]] = old_new_id_map[node]


    elif len(id_daughters) == 2:  # a division event
      new_id_ts_te_map[old_new_id_map[id_daughters[0]]][2] = old_new_id_map[
        node]  # set the parent correctly to first daughter
      new_id_ts_te_map[old_new_id_map[id_daughters[1]]][2] = old_new_id_map[
        node]  # set the parent correctly to second daughter

  import json
  with open(ilp_dirname + '/old_new_id_map.json', 'w') as fp:
    json.dump(old_new_id_map, fp)

  res_list = []
  for key in new_id_ts_te_map.keys():
    res_list.append(np.array([key, new_id_ts_te_map[key][0], new_id_ts_te_map[key][1], new_id_ts_te_map[key][2]]))

  df = pd.DataFrame(res_list)
  df.to_csv(ilp_dirname + '/res_track.txt', index=False, header=False, sep=' ')

  for t in range(len(mask_filenames)):
    ma = tifffile.imread(mask_filenames[t])
    ma_empty = np.zeros_like(ma)
    tifffile.imsave(ilp_dirname + '/mask' + str(t).zfill(3) + '.tif', ma_empty)

  for key in tqdm(old_new_id_map.keys()):
    new_id = old_new_id_map[key]
    label_last = np.zeros_like(ma)
    for t in range(new_id_ts_te_map[new_id][0], new_id_ts_te_map[new_id][1] + 1):
      ma_ilp = tifffile.imread(ilp_dirname + '/mask' + str(t).zfill(3) + '.tif')
      ma = tifffile.imread(tracklets_dirname + '/mask' + str(t).zfill(3) + '.tif')
      label_key=(ma==int(key))
      all_keys =[k for k, v in old_new_id_map.items() if v == old_new_id_map[key]]
      s = 0
      for key_it in all_keys:
        s += (ma==int(key_it)).sum()
      if s>0:
        ma_ilp[label_key] = new_id
        label_last = label_key
      else:
        ma_ilp[label_last] = new_id
      tifffile.imsave(ilp_dirname + '/mask' + str(t).zfill(3) + '.tif', ma_ilp)

  # execute the bash script
  if save_results:
    os.rename(ilp_dirname, os.path.join(save_dir, project_name + '_RES/'))
    subprocess.check_call(["../../../LineageTracer/utils/TRAMeasure", save_dir, project_name, '3'], shell=False)
    os.rename(os.path.join(save_dir, project_name + '_RES/'), ilp_dirname)
