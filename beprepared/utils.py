import os
import shutil

def copy_or_hardlink(src, dst):
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError) as e:
        shutil.copy2(src, dst)

def shorten_path(path, max_length=60):
    """
    Shorten a file path to a maximum length by inserting an ellipsis in the middle if needed.

    Args:
        path (str): The original file path.
        max_length (int): The maximum allowed length of the shortened path.

    Returns:
        str: The shortened file path with an ellipsis in the middle if necessary.
    """
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not isinstance(max_length, int) or max_length < 5:
        raise ValueError("max_length must be an integer greater than or equal to 5")

    if len(path) <= max_length:
        return path
    else:
        # Calculate the number of characters to show on each side
        # Reserve 3 characters for the ellipsis
        chars_each_side = (max_length - 3) // 2
        # If max_length is odd, add one more character to the front part
        front = path[:chars_each_side + (max_length - 3) % 2]
        back = path[-chars_each_side:]
        return f"{front}...{back}"

