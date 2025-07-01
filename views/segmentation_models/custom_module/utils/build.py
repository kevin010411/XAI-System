from mmengine import Registry

MODELS = Registry("model")
TRANSFORMS = Registry("transform")


def build_model(cfg):
    """Build a model from the given configuration.

    Args:
        cfg (Config): The configuration object containing model settings.

    Returns:
        nn.Module: The constructed model.
    """
    return MODELS.build(cfg)


def build_transform(cfg):
    """Build a transform from the given configuration.

    Args:
        cfg (Config): The configuration object containing transform settings.

    Returns:
        nn.Module: The constructed transform.
    """
    return TRANSFORMS.build(cfg)
