# call c++ exe as:
# ./bending_models/punched_disk_initial_curved -t 1e-2 -a 1e-3 -s 2 -m 1

import os
import subprocess
import argparse

binary_path = '/Users/zhenc/Projects/libshell/build/bending_models/result_comp'

material_types = [
    1 # neohookean
]

# exe -r results_dir
def run_analysis(output_dir):
    for material_type in material_types:
        material_name = "Neohookean" if material_type == 1 else "StVK"
        results_dir = os.path.join(output_dir, material_name)

        # Loop all the sub folders in the results_dir
        for sub_dir in os.listdir(results_dir):
            if os.path.isdir(os.path.join(results_dir, sub_dir)):
                actual_results_dir = os.path.join(results_dir, sub_dir)
                arguments = [binary_path, 
                '-r', actual_results_dir]
                try:
                    print(arguments)
                    result = subprocess.run(arguments, check=True, capture_output=True, text=True)
                    # print("stdout:", result.stdout)
                    # print("stderr:", result.stderr)
                except subprocess.CalledProcessError as e:
                    print(f"Error occurred: {e}") 
    
           
def main(output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    run_analysis(output_dir)

if __name__ == "__main__":
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default='./output')
    args = parser.parse_args()
    main(args.output_dir)  






