import os
import shutil
import random
import tqdm
import ollama


output_path = "data/0.ilsvrc"
dataset_path = "/work/pi_juanzhai_umass_edu/gehaozhang/ILSVRC/Data/DET/test/"
select_count = 90

if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path, exist_ok=True)

file_names = os.listdir(dataset_path)

file_names_had_run = """
ILSVRC2017_test_00000978.JPEG
ILSVRC2017_test_00001680.JPEG
ILSVRC2017_test_00002166.JPEG
ILSVRC2017_test_00002850.JPEG
ILSVRC2017_test_00003972.JPEG
ILSVRC2017_test_00001479.JPEG
ILSVRC2017_test_00001869.JPEG
ILSVRC2017_test_00002435.JPEG
ILSVRC2017_test_00003842.JPEG
ILSVRC2017_test_00005422.JPEG
"""

file_names = list(set(file_names) - set(file_names_had_run.strip().split("\n")))

file_names_selected = random.sample(file_names, select_count)

for file_name in file_names_selected:
    
    file_path = os.path.join(dataset_path, file_name)
    shutil.copy(
        os.path.join(dataset_path, file_name),
        os.path.join(output_path, file_name)
    )