import pickle


def read_split(split_path):
    with open(split_path, 'rb') as handle:
        file = pickle.load(handle)
    lst = []
    for item in file:
        lst.append(item)
    return lst


def read_patient_labels(patient_list_path):
    file = open(patient_list_path)
    patient_labels = {}
    for line in file:
        line = line.strip()
        line = line.split(',')
        patient_id = line[0]
        label = int(line[2])
        patient_labels[patient_id] = label
    return patient_labels


def str_to_dict(arg_str):
    """Convert a string to a dictionary."""
    mydict = dict(item.split(":") for item in arg_str.split(",") if ":" in item)
    return mydict