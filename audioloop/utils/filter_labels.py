import argparse
import csv
import os

DEFAULT_CSV_PATH = "data/urbansound8k/UrbanSound8k.csv"
DEFAULT_DATA_DIR = "data/urbansound8k"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", help="path to csv metadata file", default=DEFAULT_CSV_PATH
    )
    parser.add_argument("--data-dir", help="path to data directory", default=DEFAULT_DATA_DIR)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--classID", help="class ID to return", nargs="*", type=int)
    group.add_argument("--classname", help="class name to return", nargs="*")
    group.add_argument("--invert", help="invert the filter", action="store_true")

    return parser.parse_args()


def get_matching_paths(args):
    with open(args.csv) as csvfile:
        reader = csv.DictReader(csvfile)
        paths = []

        for row in reader:
            filename = fold = None
            match = False

            if args.classID:
                match = int(row["classID"]) in args.classID
            elif args.classname:
                match = row["class"] in args.classname
            else:
                match = True

            if args.invert:
                match = not match

            if match:
                filename = row["slice_file_name"]
                fold = row["fold"]
                paths.append(
                    os.path.join(os.getcwd(), args.data_dir, f"fold{fold}", filename)
                )
    return paths


def get_matching_paths_from_csv(csv_path=DEFAULT_CSV_PATH, data_dir=DEFAULT_DATA_DIR, classnames=None, classIDs=None, invert=False):
    args = argparse.Namespace(
        csv=csv_path, data_dir=data_dir, classname=classnames, classID=classIDs, invert=invert
    )
    return get_matching_paths(args)


def main():
    args = parse_args()
    paths = get_matching_paths(args)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
