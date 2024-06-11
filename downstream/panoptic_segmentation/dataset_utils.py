import yaml
import numpy as np


def collate_dataset_info(dataset_name):
    dataset_type = dataset_name.lower()
    if dataset_type == 'SemanticKitti':
        with open("./config/dataset/semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        SemKITTI_label_name = dict()
        for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
            SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]
        unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
        unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

        thing_class = semkittiyaml['thing_class']
        thing_list = [cl for cl, ignored in thing_class.items() if ignored]

        return unique_label, unique_label_str, thing_list
    elif dataset_type == 'nuscenes':
        with open("./config/dataset/nuscenes.yaml", 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        nuscenes_label_name = nuscenesyaml['labels_16']
        unique_label = np.asarray(sorted(list(nuscenes_label_name.keys())))[1:] - 1
        unique_label_str = [nuscenes_label_name[x] for x in unique_label + 1]

        thing_class = nuscenesyaml['thing_class']
        thing_list = [cl for cl, ignored in thing_class.items() if ignored]

        return unique_label, unique_label_str, thing_list
    else:
        raise NotImplementedError
