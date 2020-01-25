from typing import List
import os
import re


def create_output_dir(dir_name) -> bool:
    if not os.path.isdir(dir_name) or not os.path.exists(dir_name):
        print("Creating output directory: %s" % dir_name)
        try:
            os.makedirs(dir_name)
        except OSError:
            print("Creation of the directory %s failed" % dir_name)
            return False
        else:
            print("Successfully created the directory %s " % dir_name)
            return True
    else:
        return True


def get_files_from_dir(dir_path: str, file_type: str = None) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    file_paths = [
        f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
    ]
    if file_type is not None:
        file_paths = [f for f in file_paths if f.lower().endswith(file_type.lower())]
    return file_paths


def _int_string_sort(file_name) -> int:
    match = re.match("[0-9]+", file_name)
    if not match:
        return 0
    return int(match[0])


def get_highest_numbered_file(
    dir_path: str, file_extention: str = None, filter_keyword=None
) -> str:
    file_names = get_files_from_dir(dir_path)

    if file_extention is not None:
        file_names = [
            file_name
            for file_name in file_names
            if file_name.lower().endswith(file_extention.lower())
        ]
    if filter_keyword is not None:
        file_names = [
            file_name
            for file_name in file_names
            if filter_keyword.lower() in file_name.lower()
        ]
    if len(file_names) == 0:
        return None
    highest_numbered_file = sorted(file_names, key=_int_string_sort, reverse=True)[0]
    return os.path.join(dir_path, highest_numbered_file)
