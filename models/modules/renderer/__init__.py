def load_renderer(cfg, canonical_info):
    name = cfg.name
    if name == 'gaussian':
        from .gaussian import Renderer
        module = Renderer(cfg, canonical_info)
    elif name == 'mesh':
        from .mesh import Renderer
        module = Renderer(cfg, canonical_info)
    else:
        raise NotImplementError(f'renderer {name} not implemented')
    return module