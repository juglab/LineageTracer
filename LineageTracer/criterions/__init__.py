from LineageTracer.criterions.lineage_tracer_loss import LineageTracerLoss

def get_loss(loss_opts):
  return LineageTracerLoss(loss_opts)
