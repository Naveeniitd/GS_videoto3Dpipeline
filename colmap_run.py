#!/usr/bin/env python3

import argparse
import os
import subprocess

def run_command(cmd: str):
    """
    Utility function to run a shell command, printing it for clarity.
    Raises an exception if the command fails.
    """
    print(f"\n[Running command]: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")

def main():
    parser = argparse.ArgumentParser(
        description="Run a COLMAP sequential reconstruction on a 360° set of images."
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Path to the directory containing your 360 car images."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory where the COLMAP database and reconstruction data will be saved."
    )
    parser.add_argument(
        "--use_gpu", type=int, default=1,
        help="Set to 1 to enable GPU for SIFT and matching, 0 to disable."
    )
    parser.add_argument(
        "--sift_max_features", type=int, default=8192,
        help="Max number of SIFT features to extract per image."
    )
    parser.add_argument(
        "--conf_thresh", type=float, default=0.5,
        help="Confidence threshold for feature matching."
    )
    parser.add_argument(
        "--nms_thresh", type=float, default=0.5,
        help="NMS threshold for feature matching."
    )
    parser.add_argument(
        "--loop_detection", type=int, default=1,
        help="Set to 1 to enable loop detection in sequential matcher, 0 to disable."
    )
    parser.add_argument(
        "--loop_period", type=int, default=30,
        help="Number of frames between loop detections. (Only used if loop_detection=1)"
    )
    parser.add_argument(
        "--mapper_init_min_tri_angle", type=float, default=4.0,
        help="Minimum triangulation angle in degrees for initial mapping."
    )
    parser.add_argument(
        "--mapper_multiple_models", type=int, default=0,
        help="Allow multiple disconnected reconstructions if set to 1."
    )
    parser.add_argument(
        "--skip_dense", action="store_true",
        help="Skip the optional dense reconstruction steps if specified."
    )
    args = parser.parse_args()

    image_dir = os.path.abspath(args.image_dir)
    output_dir = os.path.abspath(args.output_dir)
    database_path = os.path.join(output_dir, "database.db")
    sparse_dir = os.path.join(output_dir, "sparse")
    dense_dir = os.path.join(output_dir, "dense")

    # Create the output directories if not exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    if not args.skip_dense:
        os.makedirs(dense_dir, exist_ok=True)

    # ----------------------------------------------------------------------------
    # 1) Feature Extraction
    # ----------------------------------------------------------------------------
    cmd_feature_extractor = (
        "colmap feature_extractor "
        f"--database_path \"{database_path}\" "
        f"--image_path \"{image_dir}\" "
        f"--SiftExtraction.use_gpu {args.use_gpu} "
        f"--SiftExtraction.max_num_features {args.sift_max_features}"
    )
    run_command(cmd_feature_extractor)

    # ----------------------------------------------------------------------------
    # 2) Sequential Image Matching
    # ----------------------------------------------------------------------------
    # We use sequential_matcher for frames in a circular (360°) sequence
    # so they can match to neighbors + loop closure between the first/last frames.
    cmd_sequential_matcher = (
        "colmap sequential_matcher "
        f"--database_path \"{database_path}\" "
        f"--SiftMatching.use_gpu {args.use_gpu} "
        "--SiftMatching.guided_matching 1 "
        f"--SequentialMatching.loop_detection {args.loop_detection} "
        f"--SequentialMatching.loop_detection_period {args.loop_period}"
    )
    run_command(cmd_sequential_matcher)

    # ----------------------------------------------------------------------------
    # 3) Mapper (Sparse Reconstruction)
    # ----------------------------------------------------------------------------
    cmd_mapper = (
        "colmap mapper "
        f"--database_path \"{database_path}\" "
        f"--image_path \"{image_dir}\" "
        f"--output_path \"{sparse_dir}\" "
        f"--Mapper.init_min_tri_angle {args.mapper_init_min_tri_angle} "
        f"--Mapper.multiple_models {args.mapper_multiple_models} "
        "--Mapper.ba_global_pba 1 "  # Parallel BA for speed with large sets
    )
    run_command(cmd_mapper)

    # ----------------------------------------------------------------------------
    # 4) Dense Reconstruction (Optional unless --skip_dense is provided)
    # ----------------------------------------------------------------------------
    if not args.skip_dense:
        # 4a) Image Undistortion
        cmd_undistort = (
            "colmap image_undistorter "
            f"--image_path \"{image_dir}\" "
            f"--input_path \"{sparse_dir}/0\" "    # Typically the first sub-folder
            f"--output_path \"{dense_dir}\" "
            "--output_type COLMAP "
            "--max_image_size 2000 "
        )
        run_command(cmd_undistort)

        # 4b) Patch Match Stereo
        cmd_patch_match = (
            "colmap patch_match_stereo "
            f"--workspace_path \"{dense_dir}\" "
            "--workspace_format COLMAP "
            "--PatchMatchStereo.gpu_index 0 "
            "--PatchMatchStereo.window_radius 5 "
        )
        run_command(cmd_patch_match)

        # 4c) Stereo Fusion
        cmd_stereo_fusion = (
            "colmap stereo_fusion "
            f"--workspace_path \"{dense_dir}\" "
            "--workspace_format COLMAP "
            "--input_type geometric "
            f"--output_path \"{dense_dir}/fused.ply\" "
        )
        run_command(cmd_stereo_fusion)

    print("\n=== COLMAP Sequential Reconstruction Completed Successfully! ===")

if __name__ == "__main__":
    main()