# Dataset preparation

- How to generate datasets?
- What is structure of datasets?
### dataset-v11-resize224x224FromV7
- 在`dataset-v7-rename_*_5`基础上，使用[step11_resize224x224FromV7.py](preprocess/step11_resize224x224FromV7.py)将原始的`1024x380`尺寸`pad`为`1024x1024`，`pad`部分为`0`在`resize`到`224x224`，因为大多数时候使用的输入图像的尺寸都是`224`，
  所以直接在数据集层面作出一个这样尺寸的数据集，可使本来在`transform`阶段做的`crop`和`resize`操作免除，以提高训练速度。实验表明，在更改之前，在`mobilenet_v2`
  运行在两个`Titan Xp`上的单`epoch`训练时间为`11.3s`，使用该数据集后时间为`8.7s`，降低`23%`。
- 数据集结构同v7
### dataset-v10-crop1024x1024
- 在`dataset-v9-convert2jpgFromv5`基础上，使用[step10_crop1024x1024.py](preprocess/step10_crop1024x1024.py)将原始的`1024x1280`尺寸`crop`为`1024x1024`，以方便在使用`tf.keras.preprocessing.image_dataset_from_directory()`API读取数据时，
  可以直接使用其内置的`image_size`参数来进行`resize`操作
- 数据集结构同v9
### dataset-v9-convert2jpgFromv5
- 在`dataset-v5-splitTrainVal`基础上，使用[step9_convert2jpgFromv5.py](preprocess/step9_convert2jpgFromv5.py)用`OpenCV`将图片由`bmp`格式转换为`jpg`格式，与v8不同,该版本是从未经crop的v5版本转换而来，
  因为`tf.keras.layers.experimental.preprocessing.CenterCrop`不支持`crop`后的尺寸比之前的尺寸更大
- 数据集结构同v5
### dataset-v8-convert2jpg
- 在`dataset-v7-rename_*_5`基础上，使用[step8_convert2jpg.py](preprocess/step8_convert2jpg.py)用`OpenCV`将图片由`bmp`格式转换为`jpg`格式，为了在`TensorFlow`中使用，
  否则使用`tf.keras.preprocessing.image_dataset_from_directory()`API读进来的数据集，在送进网络中时，总是报通道数不匹配的错误，
  将`bmp`格式读进来的图片全部识别成了单通道的，转换为`jpg`格式之后，不再报错。 
- 数据集结构同v7
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
