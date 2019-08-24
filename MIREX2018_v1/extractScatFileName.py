import os


def file_name(file_dir):
    for _, _, files in os.walk(file_dir):
        return files


def get_feature_files(file_dir):
    res = []
    with open(file_dir, "r") as f:
        lines = f.readlines()
        for i in lines:
            res.append(i)
    return res


# files = file_name("../scratch_folder/preprocessing/scat_coefficients")
# files = get_feature_files("./debug_data/featureExtractionListFile.txt")
train_file = "./debug_data/trainListFile.txt"
test_file = "./debug_data/testListFile.txt"
feature_file = "./debug_data/featureExtractionListFile.txt"

with open(train_file, 'w') as f:
    with open(test_file, 'w') as g:
        files = get_feature_files("./debug_data/featureExtractionListFile.txt")
        for i in files:
            name = i
            if "00009" not in name and "00008" not in name:
                box = name[name.rfind('/') + 1:name.find('.')]
                f.write(name.strip("\n") + '\t' + box + "\n")
            else:
                g.write(name)

# with open(feature_file, 'w') as f:
#     files = file_name("../scratch_folder/preprocessing/scat_coefficients")
#     for i in files:
#         name = i.strip(".scat").replace("-", "/")
#         f.write(name + "\n")
