import os
import multiprocessing as mp
import numpy as np
from tqdm import trange, tqdm
from pesummary.utils.utils import iterator

indices = np.arange(200000)

filenamesA = np.empty(len(indices)).astype(str)
filenamesB = np.empty(len(indices)).astype(str)
for idx in indices:
    filenamesA[idx] = f"signal{idx}_hp.gwf"
    filenamesB[idx] = f"signal{idx}_hc.gwf"

filenames = np.append(filenamesA, filenamesB)

def remove_files(filename):
    loc = "runfiles/tmp_delete/waveforms/" + filename
    if os.path.isfile(loc):
        os.remove(loc)

with mp.Pool(32) as pool:
    results = np.array(list(iterator(
        pool.imap(remove_files, filenames),
        tqdm=True,
        desc=f"Deleting files",
        total=len(filenames),
        bar_format = "{desc} | {percentage:3.0f}% | {n_fmt}/{total_fmt} | {elapsed}"
    )))


'''for filename in tqdm(filenames):
    loc = path + filename
    if os.path.isfile(loc):
    os.remove(loc)
'''
