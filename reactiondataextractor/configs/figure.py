class GlobalFigureMixin:
    """If no `figure` was passed to an initializer, use the figure stored in configs
    (set at the beginning of extraction)"""
    def __init__(self, fig):
        if fig is None:
            import configs.config
            self.fig = configs.config.Config.FIGURE
        else:
            self.fig = fig