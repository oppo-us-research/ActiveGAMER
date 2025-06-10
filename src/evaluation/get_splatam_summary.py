import os
import csv
import numpy as np

def calculate_average(filepath):
    """Calculate the average of values in a text file."""
    try:
        with open(filepath, 'r') as f:
            values = [float(line.strip()) for line in f if line.strip()]
        if values:
            return np.mean(values)
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
    return "/"

def read_eval_3d_results(filepath):
    """Read and return values from eval_3d_result.txt file."""
    results = {"acc": "/", "comp": "/", "comp%": "/"}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                key, value = line.strip().split(',')
                results[key] = float(value)
    except Exception as e:
        print(f"Error reading eval_3d file {filepath}: {e}")
    return results["acc"], results["comp"], results["comp%"]

def summarize_experiments(base_dir, experiment_list, output_csv='experiment_summary_activeslam_mp3d.csv'):
    # scenes = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
    scenes = ["GdvgFV5R1Z5", "gZ6f7yhEvPG", "HxpKQynjfin", "pLe4wQe7qrG", "YmJkqBEsHnH"]
    stages = ["eval_exploration_stage_0", "eval_exploration_stage_1", "eval_final"]
    metrics_files = ["psnr.txt", "ssim.txt", "lpips.txt", "l1.txt", "rmse.txt"]

    # Prepare CSV file
    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["scene", "exp", "stage", "psnr", "ssim", "lpips", "l1", "rmse", "acc", "comp", "comp%"])

        for scene in scenes:
            for exp in experiment_list:
                for stage in stages:
                    # Define the path to the stage folder
                    splatam_stage_path = os.path.join(base_dir, scene, exp, "run_0", "splatam", stage)
                    stage_abbr = stage.replace("eval_", "")
                    eval_3d_stage_path = os.path.join(base_dir, scene, exp, "run_0", "eval_3d", stage_abbr)

                    # Initialize row data with defaults
                    row = [scene, exp, stage_abbr]
                    metric_averages = []

                    # Check each metric file in splatam
                    for metric in metrics_files:
                        metric_path = os.path.join(splatam_stage_path, metric)
                        if os.path.exists(metric_path):
                            average_value = calculate_average(metric_path)
                        else:
                            average_value = "/"
                        metric_averages.append(average_value)

                    # Check for eval_3d results
                    eval_3d_results_path = os.path.join(eval_3d_stage_path, "eval_3d_result.txt")
                    if os.path.exists(eval_3d_results_path):
                        acc, comp, comp_percent = read_eval_3d_results(eval_3d_results_path)
                    else:
                        acc, comp, comp_percent = "/", "/", "/"

                    # Add row to CSV
                    writer.writerow(row + metric_averages + [acc, comp, comp_percent])

    print(f"Experiment summary saved to {output_csv}")

# Usage example:
base_directory = "results/MP3D"  # Adjust base directory if needed
# experiment_list = ["ActiveGS", "ActiveGSFull", "ActiveLang", "ActiveGS_noGlobal"]  # Add or modify experiments as required
experiment_list = ["ActiveGS"]  # Add or modify experiments as required
summarize_experiments(base_directory, experiment_list)
