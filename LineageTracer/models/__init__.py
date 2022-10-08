from LineageTracer.models.lineage_tracer_net import LineageTracerNet
def get_model(name, model_opts):
    if name == "tracker_net":
        model = LineageTracerNet(**model_opts)
        return model
    else:
        raise RuntimeError("model \"{}\" not available".format(name))

