import os
import tarfile


def extract_tgz(in_dir, subdir, out_dir='data'):
    with tarfile.open(in_dir, 'r:gz') as tar:
        if os.path.isdir(f'{out_dir}/{subdir}'):
            print('Files are extracted yet. Skipping.')
        else:
            print('Extracting files...')
            subdir_and_files = [
                tarinfo for tarinfo in tar.getmembers()
                if tarinfo.name.startswith(subdir)
            ]
            tar.extractall(out_dir, members=subdir_and_files)
