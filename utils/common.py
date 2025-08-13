import os


repo_root = os.getenv('PYTHONPATH')

def get_runtime_dir(name: str):
    """A dedicated directory to dump outputs on-the-fly."""
    return os.path.join(repo_root, 'runtime', name)
