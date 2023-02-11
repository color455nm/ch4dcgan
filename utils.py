import errno
import os
import shutil
import sys
import json


def load_json(fn):
    """Load a json file
    """
    with open(fn, "r") as fn:
        file=json.load(fn)
    return file


def copy_file(fn, folder_path):
    """Copy a file to folder_path
    """
    shutil.copy(fn, folder_path)


def to_np(var):
    """Exports torch.Tensor to numpy array.
    """
    return var.detach().cpu().numpy()


def create_folder(folder_path):
    """Create a folder if it does not exists.
    """
    try:
        os.makedirs(folder_path)
    except OSError as _e:
        if _e.errno != errno.EEXIST:
            raise


def clear_folder(folder_path):
    """Clear all contents recursively if the folder exists.
    Create the folder if it has been accidently deleted.
    """
    create_folder(folder_path) # does nothing if the folder exists
    for the_file in os.listdir(folder_path):
        _file_path = os.path.join(folder_path, the_file)
        try:
            if os.path.isfile(_file_path):
                os.unlink(_file_path)
                shutil.rmtree(_file_path)
        except OSError as _e:
            print(_e)


class StdOut(object):
    """Redirect stdout to file, and print to console as well.
    """
    def __init__(self, out_file):
        self.terminal = sys.stdout
        self.log = open(out_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
