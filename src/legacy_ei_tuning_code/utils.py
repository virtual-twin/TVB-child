import pandas as pd
import nibabel as nib
import os 
import numpy as np
from tvb.simulator.lab import *
import dill as pickle

def load_data(data_root, weight_fname, tracts_fname, fc_fname):
    weights = pd.read_csv(
    os.path.join(data_root, weight_fname),
    header=None,
    ).values
    tracts = pd.read_csv(
        os.path.join(data_root, tracts_fname),
        header=None,
    ).values

    # load txt
    column_names = ["Index", "Region", "Description", "R", "G", "B", "Alpha"]

    region_info = pd.read_csv(
        os.path.join(data_root, "hcpmmp1_ordered.txt"),
        sep="\s+",
        names=column_names,
        comment="#",
    )

    # Log scaled version of the SC
    conn = connectivity.Connectivity(
        tract_lengths=tracts,
        weights=np.log1p(weights),
        centres=np.array([0.0] * weights.shape[0]),
        orientations=np.array([0.0] * weights.shape[0]),
    )
    conn.region_labels = np.array(region_info["Region"], dtype=str)
    conn.configure()
    conn.weights = 0.5 * conn.scaled_weights(mode="tract")

    fc = nib.load(
        os.path.join(data_root, fc_fname)
    ).get_fdata()

    fc[np.diag_indices(fc.shape[0])] = 0
    return fc, conn

def load_data_cluster(sc_path, fc_path, subject, region_info_path = "./data/hcpmmp1_ordered.txt"):
    weights = pd.read_csv(
    os.path.join(sc_path, subject, f"sub-{subject}_V1_MR_parc-mmp1_sc_weights.csv"),
    header=None,
    ).values
    tracts = pd.read_csv(
        os.path.join(sc_path, subject, f"sub-{subject}_V1_MR_parc-mmp1_sc_lengths.csv"),
        header=None,
    ).values

    # load txt
    column_names = ["Index", "Region", "Description", "R", "G", "B", "Alpha"]

    region_info = pd.read_csv(
        region_info_path,
        sep="\s+",
        names=column_names,
        comment="#",
    )

    # Log scaled version of the SC
    conn = connectivity.Connectivity(
        tract_lengths=tracts,
        weights=np.log1p(weights),
        centres=np.array([0.0] * weights.shape[0]),
        orientations=np.array([0.0] * weights.shape[0]),
    )
    conn.region_labels = np.array(region_info["Region"], dtype=str)
    conn.configure()
    conn.weights = 0.5 * conn.scaled_weights(mode="tract")

    fc = nib.load(
        os.path.join(fc_path, subject, f"sub-{subject}_parc-hcpmmp1_desc-fc.pconn.nii")
    ).get_fdata()

    fc[np.diag_indices(fc.shape[0])] = 0
    return fc, conn

def save_results(path, p_results):
    with open(path, "wb") as f:
        pickle.dump(p_results, f)
    return None

def load_results(path):
    with open(path, "rb") as f:
        p_results = pickle.load(f)
    return p_results