# Dataset preparation

- How to generate datasets?
- What is structure of datasets?

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
- 在`dataset-v5-splitTrainVal`基础上，使用[step6_cropImg.py](preprocess/step6_cropImg.py)，将每张图片的两边分别裁掉450像素，裁剪之后的尺寸为380x1024
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
- 在`dataset-v4-mergeChannel`基础上，使用[step5_splitTrainVal.py](preprocess/step5_splitTrainVal.py)，按照训练比验证3比1的比例，分成训练集和验证集。
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
- 在`dataset-v3-addCls`基础上，使用[step41_mergeChannel.py](preprocess/step41_mergeChannel.py)，将5块，10块，20块，30块和50块这五类的不同通道数据合并到一起
- 在`dataset-v3-addCls`基础上，使用[step42_mergeProblemSamplesCls.py](preprocess/step42_mergeProblemSamplesCls.py)，将放反，两个距离过近，偏上（下）和卡籽这四类的不同子类及不同通道数据合并到一起
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
- 在`dataset-v1-CutCon`基础上，使用[step2_whited.py](preprocess/step2_whited.py)，将图片中两侧干扰物体置白，两侧均置白450像素。生成`dataset-v2-whited`
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
