import os

root = "/home/lijiachen/data/binlangDataset/"
label_dict = {
    "0_0": 0,
    "0_1": 1,
    "15_0": 2,
    "15_1": 3,
    "20_0": 4,
    "20_1": 5,
    "30_0": 6,
    "30_1": 7,
}
for path in os.listdir(root):
    file = open(root+path+"/"+"{}.txt".format(path.split("D")[0]), "a")
    for class_name in os.listdir(root + path + "/"):
        print(class_name)
        if not class_name.endswith("txt"):
            for img_name in os.listdir(root + path + "/" + class_name):
                file.write(root + path + "/" + class_name + "/" + img_name + " " + str(label_dict[class_name]))
                file.write("\n")
    file.close()
