# Dataset preparation

If you want to reproduce the results in the paper for benchmark evaluation or training, you will need to setup datasets.

### MOT 2017

MOT is is used to train and evaluate the system. We will only use the training set (and create a validation set from it) for developing this project.

We have packed the dataset preprocessing code as a script.

~~~
    cd $CenterTrack_ROOT/tools/
    bash get_mot_17.sh
~~~

The script includes:

- Download and unzip the dataset from [MOT17 website](https://motchallenge.net/data/MOT17/).
- Convert it into COCO format using `tools/convert_mot_to_coco.py`.
- Create the half-half train/ val set described in the paper.
- Convert the public detection into a specific format.
- The output data structure should be:

  ~~~
  ${CenterTrack_ROOT}
  |-- data
  `-- |-- mot17
      `-- |--- train
          |   |--- MOT17-02-FRCNN
          |   |    |--- img1
          |   |    |--- gt
          |   |    |   |--- gt.txt
          |   |    |   |--- gt_train_half.txt
          |   |    |   |--- gt_val_half.txt
          |   |    |--- det
          |   |    |   |--- det.txt
          |   |    |   |--- det_train_half.txt
          |   |    |   |--- det_val_half.txt
          |   |--- ...
          |--- test
          |   |--- MOT17-01-FRCNN
          |---|--- ...
          `---| annotations
              |--- train_half.json
              |--- val_half.json
              |--- train.json
              `--- test.json
  ~~~
  ### dataset-v7-rename_*_5
  - 在`dataset-v6-cropImg`基础上，将`*_5`类别重命名为`*_05`，以方便在`cls_map_id`时的对应<br/>
  `{"*_01": 0, "*_02": 1 ,"*_03": 2, "*_04": 3, "*_05": 4, "*_10": 5 ,"*_20": 6, "*_30": 7, "*_50": 8}`
    ~~~
    ${data_ROOT}
    |--- cut
    |    |--- train
    |    |    |--- cut_50
    |    |    |--- cut_30
    |    |    |--- cut_20
    |    |    |--- cut_10
    |    |    |--- cut_05
    |    |    |--- cut_04
    |    |    |--- cut_03
    |    |    |--- cut_02
    |    |    `--- cut_01
    |    `--- val
    `--- con
    ~~~
  ### dataset-v6-cropImg
  - `step6_cropImg.py:`<br/>
  在`dataset-v5-splitTrainVal`基础上，将每张图片的两边分别裁掉450像素，裁剪之后的尺寸为380x1024
    ~~~
    ${data_ROOT}
    |--- cut
    |    |--- train
    |    |    |--- cut_50
    |    |    |--- ...
    |    |    |--- cut_5
    |    |    |--- ...
    |    |    `--- cut_01
    |    `--- val
    `--- con
    ~~~
  ### dataset-v5-splitTrainVal
  - `step5_splitTrainVal.py:`<br/>在`dataset-v4-mergeChannel`基础上，按照训练比验证3比1的比例，分成训练集和验证集。
    ~~~
    ${data_ROOT}
    |--- cut
    |    |--- train
    |    |    |--- cut_50
    |    |    |--- ...
    |    `--- val
    `--- con
    ~~~
  ### dataset-v4-mergeChannel
  - `step41_mergeChannel.py:`<br/>在`dataset-v3-addCls`基础上，将5块，10块，20块，30块和50块这五类的不同通道数据合并到一起
  - `step42_mergeProblemSamplesCls.py:`<br/>在`dataset-v3-addCls`基础上，将放反，两个距离过近，偏上（下）和卡籽这四类的不同子类及不同通道数据合并到一起
  - 以上两个代码共同生成dataset-v4-mergeChannel
    ~~~
    ${data_ROOT}
    |--- cut
    |    |--- cut_50
    |    |--- ...
    `--- con
    ~~~
  ### dataset-v3-addCls
  - 在`dataset-v2-whited`基础上，手动挑选出放反，两个距离过近，偏上（下）和卡籽这四类，分别对应`*_01,*_02,*_03,*_04`
    ~~~
    ${data_ROOT}
    |--- cut
    |    |--- cut_50
    |    |    |--- channel_8
    |    |    |--- ...
    |    |    `--- channel_1
    |    |--- cut_30
    |    |--- cut_20
    |    |--- cut_10
    |    |--- cut_5
    |    |--- cut_04
    |    |    |--- 50
    |    |    |    |--- channel_8
    |    |    |    |--- ...
    |    |    |    `--- channel_1
    |    |    |--- 30
    |    |    |--- 20
    |    |    |--- 10
    |    |    `--- 5
    |    |--- cut_03
    |    |--- cut_02
    |    |--- cut_01
    `--- con
    ~~~
  ### dataset-v2-whited
  - `step2_whited.py:`<br/>
  在`dataset-v1-CutCon`基础上，将图片中两侧干扰物体置白，两侧均置白450像素。生成`dataset-v2-whited`
    ~~~
    ${data_ROOT}
    |--- cut
    |    |--- cut_50
    |    |    |--- channel_8
    |    |    |--- ...
    |    |--- cut_30
    |    |--- cut_20
    |    |--- cut_10
    |    |--- cut_5
    `--- con
    ~~~
  ### dataset-v1-CutCon
  - 将原始数据手动分到cut和con两个大文件夹中
    ~~~
    ${data_ROOT}
    |--- cut
    |    |--- cut_50
    |    |    |--- channel_8
    |    |    |--- ...
    |    |--- cut_30
    |    |--- cut_20
    |    |--- cut_10
    |    `--- cut_5
    `--- con
    ~~~
  ### rowData
  - 原始采集未经处理的数据
    ~~~
    ${data_ROOT}
    |--- cut_5
    |    |--- channel_8
    |    |--- ...
    |--- cut_4
    |--- cut_3
    |--- cut_2
    |--- cut_1
    |--- con_5
    |--- con_4
    |--- con_3
    |--- con_2
    `--- con_1
    ~~~

### KITTI Tracking

We use KITTI Tracking to train and evaluate the system as well. Again, we will only use the training set (and create a validation set from it) for developing this project. Note that KITTI Tracking is 2D tracking and is different from KITTI detection (they use the same image, but different train/ val set).

- Download [images](http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip), [annotations](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip), and [calibration information](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) (not used in 2D tracking, only if you want to demo 3D detection/ tracking) from [KITTI Tracking website](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) and unzip. Place or symlink the data as below:

  ~~~
  ${CenterTrack_ROOT}
  |-- data
  `-- |-- kitti_tracking
      `-- |-- data_tracking_image_2
          |   |-- training
          |   |-- |-- image_02
          |   |-- |-- |-- 0000
          |   |-- |-- |-- ...
          |-- |-- testing
          |-- label_02
          |   |-- 0000.txt
          |   |-- ...
          `-- data_tracking_calib
  ~~~

- Run `python convert_kitti_to_coco.py` in `tools` to convert the annotation into COCO format.
- The resulting data structure should look like:

  ~~~
  ${CenterTrack_ROOT}
  |-- data
  `-- |-- kitti_tracking
      `-- |-- data_tracking_image_2
          |   |-- training
          |   |   |-- image_02
          |   |   |   |-- 0000
          |   |   |   |-- ...
          |-- |-- testing
          |-- label_02
          |   |-- 0000.txt
          |   |-- ...
          |-- data_tracking_calib
          |-- label_02_val_half
          |   |-- 0000.txt
          |   |-- ...
          |-- label_02_train_half
          |   |-- 0000.txt
          |   |-- ...
          `-- annotations
              |-- tracking_train.json
              |-- tracking_test.json
              |-- tracking_train_half.json
              `-- tracking_val_half.json
  ~~~

### nuScenes

nuScenes is used for training and evaluating 3D object tracking. We also used nuScenes for pretraining KITTI models.


- Download the dataset from [nuScenes website](https://www.nuscenes.org/download). You only need to download the "Keyframe blobs", and only need the images data. You also need to download the maps and all metadata to make the nuScenes API happy.


- Unzip, rename, and place (or symlink) the data as below. You will need to merge folders from different zip files.

  ~~~
  ${CenterTrack_ROOT}
  |-- data
  `-- |-- nuscenes
      `-- |-- v1.0-trainval
          |   |-- samples
          |   |   |-- CAM_BACK
          |   |   |   | -- xxx.jpg
          |   |   |-- CAM_BACK_LEFT
          |   |   |-- CAM_BACK_RIGHT
          |   |   |-- CAM_FRONT
          |   |   |-- CAM_FRONT_LEFT
          |   |   |-- CAM_FRONT_RIGHT
          |-- |-- maps
          `-- |-- v1.0-trainval_meta
  ~~~

- Run `python convert_kitti_to_coco.py` in `tools` to convert the annotation into COCO format. It will create `train.json`, `val.json`, `test.json` under `data/nuscenes/annotations`. nuScenes API is required for running the data preprocessing.

### CrowdHuman

CrowdHuman is used for pretraining the MOT model. Only the training set is used.

- Download the dataset from [its website](https://www.crowdhuman.org/download.html).

- Unzip and place (or symlink) the data as below. You will need to merge folders from different zip files.

  ~~~
  ${CenterTrack_ROOT}
  |-- data
  `-- |-- crowdhuman
      |-- |-- CrowdHuman_train
      |   |   |-- Images
      |-- |-- CrowdHuman_val
      |   |   |-- Images
      |-- |-- annotation_train.odgt
      |-- |-- annotation_val.odgt
  ~~~

- Run `python convert_kitti_to_coco.py` in `tools` to convert the annotation into COCO format. It will create `train.json`, `val.json` under `data/crowdhuman/annotations`.

### COCO

COCO is used to train a demo system for 80-category tracking or pose tracking.
The models are NOT evaluated in any benchmarks.

- Download the images (2017 Train, 2017 Val, 2017 Test) from [coco website](http://cocodataset.org/#download).
- Download annotation files (2017 train/val and test image info) from [coco website](http://cocodataset.org/#download).
- Place the data (or create symlinks) to make the data folder like:

  ~~~
  ${CenterTrack_ROOT}
  |-- data
  `-- |-- coco
      `-- |-- annotations
          |   |-- instances_train2017.json
          |   |-- instances_val2017.json
          |   |-- person_keypoints_train2017.json
          |   |-- person_keypoints_val2017.json
          |   |-- image_info_test-dev2017.json
          |---|-- train2017
          |---|-- val2017
          `---|-- test2017
  ~~~


## References
Please cite the corresponding References if you use the datasets.

~~~
  @article{MOT16,
    title = {{MOT}16: {A} Benchmark for Multi-Object Tracking},
    shorttitle = {MOT16},
    url = {http://arxiv.org/abs/1603.00831},
    journal = {arXiv:1603.00831 [cs]},
    author = {Milan, A. and Leal-Taix\'{e}, L. and Reid, I. and Roth, S. and Schindler, K.},
    month = mar,
    year = {2016},
    note = {arXiv: 1603.00831},
    keywords = {Computer Science - Computer Vision and Pattern Recognition}
  }

  @article{shao2018crowdhuman,
    title={Crowdhuman: A benchmark for detecting human in a crowd},
    author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
    journal={arXiv:1805.00123},
    year={2018}
  }

  @INPROCEEDINGS{Geiger2012CVPR,
    author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
    title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
    booktitle = {CVPR},
    year = {2012}
  }

  @inproceedings{lin2014microsoft,
    title={Microsoft {COCO}: Common objects in context},
    author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
    booktitle={ECCV},
    year={2014},
  }

  @inproceedings{nuscenes2019,
  title={{nuScenes}: A multimodal dataset for autonomous driving},
  author={Holger Caesar and Varun Bankiti and Alex H. Lang and Sourabh Vora and Venice Erin Liong and Qiang Xu and Anush Krishnan and Yu Pan and Giancarlo Baldan and Oscar Beijbom},
  booktitle={CVPR},
  year={2020}
  }
~~~
