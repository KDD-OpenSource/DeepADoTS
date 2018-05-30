from . import Dataset


class RealDataset(Dataset):
    """

    ToDo:
        * refactor data()
        * much more
    """

    def __init__(self, raw_path, **kwargs):
        super().__init__(kwargs)
        self.raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/raw/", raw_path)
