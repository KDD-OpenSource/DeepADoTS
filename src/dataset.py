import os

class Dataset:
    def __init__(self, name, raw_path, processed_path):
        self.name = name
        self.raw_path = os.path.join(os.getcwd(), "data/raw/", raw_path)
        self.processed_path = os.path.join(os.getcwd(), "data/processed/", processed_path)