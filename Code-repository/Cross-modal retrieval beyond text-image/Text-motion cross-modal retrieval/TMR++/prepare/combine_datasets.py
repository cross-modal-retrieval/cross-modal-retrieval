import copy
import logging
import hydra
import json
import numpy as np
import os
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


SUFFIX_DICT = {"humanml3d": "h", "kitml": "k", "babel": "b"}

@hydra.main(config_path="../configs", config_name="combine_datasets", version_base="1.3")
def combine_datasets(cfg: DictConfig):

    train_datasets = cfg.datasets
    annotations_folder_path = cfg.annotations_path

    combined_dataset_name = "_".join(train_datasets)
    combined_dataset_folder = os.path.join(annotations_folder_path, combined_dataset_name)
    os.makedirs(combined_dataset_folder, exist_ok=True)

    annotations = {}
    annotations_paraphrases = {}
    annotations_actions = {}

    annotations = {}
    annotations_paraphrases = {}
    annotations_actions = {}
    annotations_all = {}

    dataset_annotations = {}

    for dataset in train_datasets:
        annotations_path = os.path.join(annotations_folder_path, dataset, "annotations.json")
        with open(annotations_path) as f:
            d = json.load(f)
        dataset_annotations[dataset] = d
        d_new = {f"{key}_{SUFFIX_DICT[dataset]}": val for key, val in d.items()}
        annotations.update(d_new)

        annotations_paraphrases_path = os.path.join(annotations_folder_path, dataset, "annotations_paraphrases.json")
        if os.path.exists(annotations_paraphrases_path):
            with open(annotations_paraphrases_path) as f:
                d = json.load(f)
            d = {f"{key}_{SUFFIX_DICT[dataset]}": val for key, val in d.items()}
            annotations_paraphrases.update(d)

        annotations_actions_path = os.path.join(annotations_folder_path, dataset, "annotations_actions.json")
        if os.path.exists(annotations_actions_path):
            with open(annotations_actions_path) as f:
                d = json.load(f)
            d = {f"{key}_{SUFFIX_DICT[dataset]}": val for key, val in d.items()}
            annotations_actions.update(d)

        annotations_all_path = os.path.join(annotations_folder_path, dataset, "annotations_all.json")
        if os.path.exists(annotations_all_path):
            with open(annotations_all_path) as f:
                d = json.load(f)
            d = {f"{key}_{SUFFIX_DICT[dataset]}": val for key, val in d.items()}
            annotations_all.update(d)
    
    with open(os.path.join(combined_dataset_folder, "annotations.json"), "w") as f:
        json.dump(annotations, f, indent=2)
    with open(os.path.join(combined_dataset_folder, "annotations_paraphrases.json"), "w") as f:
        json.dump(annotations_paraphrases, f, indent=2)
    with open(os.path.join(combined_dataset_folder, "annotations_actions.json"), "w") as f:
        json.dump(annotations_actions, f, indent=2)
    with open(os.path.join(combined_dataset_folder, "annotations_all.json"), "w") as f:
        json.dump(annotations_all, f, indent=2)

    test_datasets = cfg.test_sets

    for dataset in test_datasets:
        if dataset not in dataset_annotations:
            annotations_path = os.path.join(annotations_folder_path, dataset, "annotations.json")
            with open(annotations_path) as f:
                d = json.load(f)
            dataset_annotations[dataset] = d
    
    dataset_splits = {}

    splits = ["train", "val"]
    for dataset in train_datasets:
        if dataset not in dataset_splits:
            dataset_splits[dataset] = {}
        for split in splits:
            with open(os.path.join(annotations_folder_path, dataset, "splits", f"{split}.txt")) as f:
                str_inds = f.read()
                inds = str_inds.split("\n")
                if inds[-1] == "":
                    inds.pop(-1)
            dic_ind_path = {ind: dataset_annotations[dataset][ind]["path"] for ind in inds}
            dataset_splits[dataset][split] = dic_ind_path

    split = 'test'
    for dataset in test_datasets:
        if dataset not in dataset_splits:
            dataset_splits[dataset] = {}
        with open(os.path.join(annotations_folder_path, dataset, "splits", f"{split}.txt")) as f:
            str_inds = f.read()
            inds = str_inds.split("\n")
            if inds[-1] == "":
                inds.pop(-1)
        dic_ind_path = {ind: dataset_annotations[dataset][ind]["path"] for ind in inds}
        dataset_splits[dataset][split] = dic_ind_path
    

    to_remove = {train_dataset: {test_dataset: {"train": [], "val": []} for test_dataset in test_datasets if test_dataset != train_dataset} for train_dataset in train_datasets}

    for train_dataset in train_datasets:

        for split in ["train", "val"]:
            for train_id, train_path in dataset_splits[train_dataset][split].items():

                for test_dataset in set(test_datasets) - set([train_dataset]):

                    for test_id, test_path in dataset_splits[test_dataset]["test"].items():
                        if train_path == test_path:

                            if not cfg.filter_babel_seg:
                                if test_dataset == "babel":
                                    test_duration = float(dataset_annotations[test_dataset][test_id]["duration"])
                                    test_fragment_duration = float(dataset_annotations[test_dataset][test_id]["fragment_duration"])

                                    if not np.isclose([test_duration], [test_fragment_duration], atol=0.1, rtol=0):
                                        continue

                                if train_dataset == "babel":
                                    train_duration = float(dataset_annotations[train_dataset][train_id]["duration"])
                                    train_fragment_duration = float(dataset_annotations[train_dataset][train_id]["fragment_duration"])

                                    if not np.isclose([train_duration], [train_fragment_duration], atol=0.1, rtol=0):
                                        continue

                            train_start = float(dataset_annotations[train_dataset][train_id]["annotations"][0]["start"])
                            train_end = float(dataset_annotations[train_dataset][train_id]["annotations"][0]["end"])
                            test_start = float(dataset_annotations[test_dataset][test_id]["annotations"][0]["start"])
                            test_end = float(dataset_annotations[test_dataset][test_id]["annotations"][0]["end"])

                            if not ((train_end <= test_start) or (test_end <= train_start)):
                                to_remove[train_dataset][test_dataset][split].append(train_id)
    
    datasets_curated = {train_dataset: {split: list(dataset_splits[train_dataset][split].keys()) for split in ["train", "val", "test"]} for train_dataset in train_datasets}
    
    for train_dataset in train_datasets:
        for test_dataset in set(test_datasets) - set([train_dataset]):
            for split in ["train", "val"]:
                for keyid in to_remove[train_dataset][test_dataset][split]:
                    if keyid in datasets_curated[train_dataset][split]:
                        datasets_curated[train_dataset][split].remove(keyid)
    
    splits_folder = os.path.join(annotations_folder_path, combined_dataset_name, "splits")
    os.makedirs(splits_folder, exist_ok=True)
    all_ids = []
    for split in ["train", "val", "test"]:
        ids = []
        for train_dataset in train_datasets:
            ids += [f'{elt}_{SUFFIX_DICT[train_dataset]}' for elt in datasets_curated[train_dataset][split]]
        all_ids += ids
        ids_str = "\n".join(ids)
        if split == "test":
            filename = f"{split}.txt"
        else:
            filename = f"{split}{cfg.split_suffix}.txt"
        with open(os.path.join(splits_folder, filename), "w") as f:
            f.write(ids_str)
    
    all_ids_str = "\n".join(all_ids)
    with open(os.path.join(splits_folder, f"all{cfg.split_suffix}.txt"), "w") as f:
        f.write(all_ids_str)


if __name__ == "__main__":
    combine_datasets()

