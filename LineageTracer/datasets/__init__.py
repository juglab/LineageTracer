from LineageTracer.datasets.lineage_tracer_dataset import LineageTracerTestDataset
from LineageTracer.datasets.lineage_tracer_dataset import LineageTracerTrainValDataset


def get_dataset(type, dataset_opts):
  if type == 'test':
    return LineageTracerTestDataset(**dataset_opts)
  else:
    return LineageTracerTrainValDataset(**dataset_opts)
