# call c++ exe as:
# ./bending_models/punched_disk_initial_curved -t 1e-2 -a 1e-3 -s 2 -m 1

import os
import subprocess
import argparse

binary_path = '/Users/zhenc/Projects/libshell/build/bending_models/punched_disk_initial_curved'

test_thickness = [
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    2e-1,
    3e-1,
    4e-1,
    5e-1,
    0.75,
    1,
]

test_thickness = [
    2e-2,
    3e-2,
    4e-2,
    5e-2,
    6e-2,
    7e-2,
    8e-2,
    9e-2
]

test_triangle_area = [
    2e-3,
    1e-3,
    5e-4,
    2e-4,
    1e-4,
    5e-5
]

test_sff_type = [
    0,          # s1 sin
    2,          # s2 sin
]

material_types = [
    1 # neohookean
]

# exe -t thickness -a triangle_area -s sff_type
def run_test(output_dir):
    for thickness in test_thickness:
        for triangle_area in test_triangle_area:
            for sff_type in test_sff_type:
                for material_type in material_types:
                    arguments = [binary_path, 
                    '-t', str(thickness),
                    '-a', str(triangle_area),
                    '-s', str(sff_type),
                    '-m', str(material_type),
                    '-o', output_dir]

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
    run_test(output_dir)

if __name__ == "__main__":
    # add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default='./output')
    args = parser.parse_args()
    main(args.output_dir)  






