import os

def copy_or_hardlink(src, dst):
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError) as e:
        shutil.copy2(src, dst)

