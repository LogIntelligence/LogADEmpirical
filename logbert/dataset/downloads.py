import subprocess
from argparse import ArgumentParser
import os

def download_dataset(dataset_name, data_dir):
    # subprocess.run(f"if [ -e {data_dir} ] then echo'{data_dir} exists' else mkdir -p {data_dir} fi")
    if dataset_name == "hdfs_2k":
        subprocess.run(f"cd  {data_dir}; wget https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log -P {data_dir}")

    elif dataset_name == "hdfs":
        zipfile = "HDFS_1.tar.gz?download = 1"
        subprocess.run(f"wget https://zenodo.org/record/3227177/files/{zipfile} -P {data_dir}; tar -xvzf {zipfile}")

    elif dataset_name == "bgl_2k":
        subprocess.run(f"cd {data_dir}; wget https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log")

    elif dataset_name == "bgl":
        zipfile = "BGL.tar.gz?download=1"
        subprocess.run(f"https://zenodo.org/record/3227177/files/{zipfile} -P{data_dir}; tar -xvzf {zipfile}")

    ##tbird


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", help="which dataset to use", choices=["hdfs", "bgl", "tbird", "hdfs_2k", "bgl_2k"])
    parser.add_argument("--data_dir", default="../../data/", metavar="DIR", help="data directory")
    parser.add_argument("--folder", help="which folder dataset belongs to")
    args = parser.parse_args('--dataset_name bgl_2k --folder bgl/'.split())

    # print(args.data_dir, os.path.exists(args.data_dir))
    # if not os.path.exists(args.data_dir):
    #     os.makedirs(args.data_dir)

    args.data_dir = os.path.join(args.data_dir,args.folder)
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    print(args.data_dir, os.path.exists(args.data_dir))


    download_dataset(args.dataset_name, args.data_dir)
