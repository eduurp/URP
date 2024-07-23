import datetime
import time
import os
import json

import numpy as np

def create_directory(directory_name):
    try:
        # make directory
        os.makedirs(directory_name)
        print(f"Directory successfully created : {directory_name}")
    except FileExistsError:
        print(f"Directory already existed : {directory_name}")
    except Exception as e:
        print(f"Erorr occured : {e}")
