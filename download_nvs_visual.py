

"""
Script to download NVS visuals from remote server.
Downloads reference views and prediction videos for PICKED_SCENES.
"""

import os
import subprocess
from pathlib import Path

# Remote server configuration
REMOTE_USER = "yuwu3"
REMOTE_HOST = "grogu.ri.cmu.edu"
REMOTE_SERVER = f"{REMOTE_USER}@{REMOTE_HOST}"

# Remote paths (from create_vis_comp.py)
LOG_ROOT = "/grogu/user/yuwu3/rayrope_log_Feb/"
MODEL_DIR = "L6-H8-D1152-FF1024-B8"

# Local output directory
LOCAL_OUTPUT_DIR = "assets/nvs-visual"

# Methods and their experiment names
MAIN_COMP_EXPS = {
    'Plucker': 'none-plucker-seed1',
    'GTA': 'gta-none-seed1',
    'PRoPE': 'prope-seed1',
    'RayRoPE': 'd_pj+0_3d-predict_dsig-inv_d-seed1',
}

# Dataset-specific input views configuration
DATASET_INPUT_VIEWS = {
    "re10k": 2,
    "objaverse": 2,
    "co3d_seen": 4,
}

PICKED_SCENES = {
    "re10k": [
        ('002ae53df0e0afe2', 1),
        ('02ee66b3efbf3b0a', 1),
        ('0131c9aed0fb3940', 1),
        ('040a26b288e7bda4', 1),
        ('04e2be0415136fa9', 0),
        ('03a78406de1d0993', 1),
        ('0068e97c1c1f61aa', 0),
        ('040895d45bf4e580', 0),

        # ('04e2be0415136fa9', 1),
        # ('03a78406de1d0993', 2),
        # ('040a26b288e7bda4', 1),
        # ('02ee66b3efbf3b0a', 1),
        # ('01a2277ee817b310', 1),
        # ('002ae53df0e0afe2', 1),
        # ('01a5cc3805e94c21', 2),
    ],
    "objaverse": [
        # radial
        ('015c200ce786438c8e35ddf635d1e236', 4),
        ('02ca74ef6a1b4ec386c11048603f0e98', 4),
        ('03c0260373c7406ea408c2dec9f8d502', 4),
        ('047ca62b79d140d4b64044db311561d9', 8),
        # spherical
        ('035d9ce9964b42f6bc20514853934d1b', 0),
        ('0050f76a07fa43b7a38e6cef40beb69d', 2),
        ('03e4ff99ecc24e5f8ea49d4e8df876e8', 2),
        ('006373e3885b472cb5538fc570235fcf', 8),
        ('002aec05c41342dea61828f67d340d2d', 8)
    ],
    "co3d_seen": [
        # 4 views for video
        ('256_27676_55062', 2),
        ('118_13853_28129', 0),
        ('112_13308_25702', 0),
        ('187_20181_35751', 2),
        ('164_17988_33493', 3),
        ('250_26773_54489', 1),
        ('222_23409_48576', 3),
    ],
}


def get_log_dir(dataset, exp_name):
    """Construct remote log directory path for an experiment."""
    return os.path.join(LOG_ROOT, MODEL_DIR, dataset, "unknown_d", exp_name)


def get_tests_subdir(input_views):
    """Get the tests subdirectory name based on input views."""
    if input_views == 2:
        return 'tests'
    else:
        return f'eval-context{input_views}'


def scp_download(remote_path, local_path, is_dir=False):
    """Download file or directory from remote server using scp."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    remote_full = f"{REMOTE_SERVER}:{remote_path}"
    cmd = ["scp", "-r" if is_dir else "", remote_full, local_path]
    cmd = [c for c in cmd if c]  # Remove empty strings
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"  ✓ Downloaded: {os.path.basename(local_path)}")
            return True
        else:
            print(f"  ✗ Failed: {remote_path}")
            print(f"    Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout: {remote_path}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def download_scene_visuals(dataset, scene, view_id, input_views, index):
    """Download reference views and prediction videos for a scene.
    
    Args:
        dataset: Dataset name (re10k, objaverse, co3d_seen)
        scene: Scene name/id
        view_id: View ID for this scene
        input_views: Number of input/reference views
        index: Index of this scene in PICKED_SCENES for the dataset
    """
    tests_subdir = get_tests_subdir(input_views)
    
    # Create output directory
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    
    print(f"\nDownloading scene: {scene} (index: {index}, view_id: {view_id})")
    
    # Use PRoPE directory as reference for common files (ref views, GT)
    prope_dir = get_log_dir(dataset, MAIN_COMP_EXPS['PRoPE'])
    remote_scene_dir = os.path.join(prope_dir, tests_subdir, scene)
    
    # Local file naming: {dataset}_{index}_{name}
    def local_path(name):
        return os.path.join(LOCAL_OUTPUT_DIR, f"{dataset}_{index}_{name}")
    
    # Download reference views
    for ref_idx in range(input_views):
        remote_ref = os.path.join(remote_scene_dir, f'ref{ref_idx}.png')
        scp_download(remote_ref, local_path(f'ref{ref_idx}.png'))
    
    # Download GT video
    remote_gt_video = os.path.join(remote_scene_dir, 'gt.mp4')
    scp_download(remote_gt_video, local_path('gt.mp4'))
    
    # Download prediction videos for each method
    for method, exp_name in MAIN_COMP_EXPS.items():
        method_dir = get_log_dir(dataset, exp_name)
        remote_pred = os.path.join(method_dir, tests_subdir, scene, 'pred.mp4')
        scp_download(remote_pred, local_path(f'{method}_pred.mp4'))


def main():
    """Main function to download all visuals."""
    # Control which datasets to download
    DOWNLOAD_RE10K = True
    DOWNLOAD_OBJAVERSE = False
    DOWNLOAD_CO3D_SEEN = False
    
    print("=" * 60)
    print("Downloading NVS Visuals from Remote Server")
    print(f"Server: {REMOTE_SERVER}")
    print(f"Output: {LOCAL_OUTPUT_DIR}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    
    # Dataset download flags
    dataset_flags = {
        "re10k": DOWNLOAD_RE10K,
        "objaverse": DOWNLOAD_OBJAVERSE,
        "co3d_seen": DOWNLOAD_CO3D_SEEN,
    }
    
    # Download for each dataset
    for dataset, scenes in PICKED_SCENES.items():
        if not dataset_flags.get(dataset, False):
            print(f"\nSkipping dataset: {dataset}")
            continue
            
        input_views = DATASET_INPUT_VIEWS.get(dataset, 2)
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset} (input_views: {input_views})")
        print(f"{'=' * 60}")
        
        for index, (scene, view_id) in enumerate(scenes):
            download_scene_visuals(dataset, scene, view_id, input_views, index)
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()