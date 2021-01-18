# Dataset preprocess
## Step 1
将原始数据手动分到cut和con两个大文件夹中
## Step 2
step2_whited.py:
在dataset-v1-CutCon基础上，将图片中两侧干扰物体置白，两侧均置白450像素。生成dataset-v2-whited
## Step 3
在dataset-v2-whited基础上，手动挑选出放反，两个距离过近，偏上（下）和卡籽这四类，分别对应*_01,*_02,*_03,*_04
## Step 4
step41_mergeChannel.py:
在dataset-v3-addCls基础上，将5块，10块，20块，30块和50块这五类的不同通道数据合并到一起
step42_mergeProblemSamplesCls.py:
在dataset-v3-addCls基础上，将放反，两个距离过近，偏上（下）和卡籽这四类的不同子类及不同通道数据合并到一起
以上两个代码共同生成dataset-v4-mergeChannel
## Step 5
step5_splitTrainVal.py:
在dataset-v4-mergeChannel基础上，按照训练比验证3比1的比例，分成训练集和验证集。生成dataset-v5-splitTrainVal
## Step 6
step6_cropImg.py:
在dataset-v5-splitTrainVal基础上，将每张图片的两边分别裁掉450像素，裁剪之后的尺寸为380x1024
  ~~~
  ${data_ROOT}
      `-- dataset-v7-rename_*_5
           |--- cut
           |    |--- train
           |    `--- val
           `--- con
                |--- train
                `--- val
      |-- dataset-v6-cropImg
      |-- dataset-v5-splitTrainVal
      |-- dataset-v4-mergeChannel
      |-- dataset-v3-addCls
      |-- dataset-v2-whited
      |-- dataset-v1-CutCon
      |-- rowData
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