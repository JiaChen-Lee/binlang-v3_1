## What's News
### Feb 13, 2021
- Result on 
  - Model: mobilenet_v3_large 
  - Input size: 224x224  
  - GPU: Titan Xp 
  - Dataset: dataset-v11
    
    |Batch Size|GPU number  |sing gpu mem|CPU util v11|Epoch time|
    |:--------:|:----------:|:----------:|:----------:|:--------:|
    |  16      |1           |2237M       |19.5%       |21.5s     |
    |  32      |1           |3653M       |20.5%       |13.6s     |
    |  64      |1           |6553M       |20.5%       |12.1s     |
    | 128      |2           |6657M       |24.5%       | 8.2s     |
- 因为batch size为16和32时的显存总量加起来也不超过一张卡的总显存，故考虑将这两个模型运行在一张卡上，测试上表所列指标知否有变化
- 经测试得出：显存不变，但是少数时间峰值会达到100%；CPU使用量也未达到100%，但是Epoch time分别变为32.5s，23.1s
- 原因分析：当两个模型运行在一张卡上时，CPU使用量未达到100%，说明Epoch time的瓶颈此时已经不是CPU，由显存峰值达到100%，推测Epoch time的增加与
  两个模型在一张卡上进行显存处理时的排队有关
### Feb 08, 2021
- Result on 
  - Model: mobilenet_v2 
  - Input size: 224x224 
  - GPU: 4 Titan Xp 
    
    |Batch Size|Video memory|Growth rate|Epoch time|Growth rate|
    |:--------:|:----------:|:---------:|:--------:|:---------:|
    |  16      |1113M       |None       |37.40s    |None       |
    |  32      |1525M       |37%        |19.60s    |-48%       |
    |  64      |2435M       |60%        |12.80s    |-35%       |
    | 128      |4271M       |75%        |11.30s    |-12%       |
    | 256      |7873M       |84%        |11.30s    |  0%       |
    | 512      |out memory  |None       |None      |None       |
- When the three mobilenet_v2 (batch size 32, 64, 128) were trained on 4 gpus simultaneously, the "epoch time" became 51s, 28s, and 22.5s respectively
- 说明同时运行多个模型，并没有获得理论上的并行加速，即使是在显存还足够的情况下
- TODO: 尝试将不同模型运行在不同GPU上，测试是否可以获得加速

- Result on:
  - Model: mobilenet_v2 
  - Input size: 224x224 
  - GPU: Titan Xp 
    
    |Batch Size|Gpu number  |sing gpu mem|Growth rate|Epoch time|Growth rate|
    |:--------:|:----------:|:----------:|:---------:|:--------:|:---------:|
    |  16      |1           |2361M       |None       |18.10s    |None       |
    |  32      |1           |4175M       |37%        |16.20s    |-48%       |
    |  64      |1           |7627M       |60%        |15.40s    |-35%       |
    | 128      |2           |7737M       |75%        |11.30s    |-12%       |
- When the three mobilenet_v2 (batch size 32, 64, 128) were trained on 1, 1, 2 gpu simultaneously, the "epoch time" became 28.5s, 28.5s, 28.5s, respectively
- It shows that even if different models are run on different GPUs, they still cannot get acceleration. It is supposed that the problem is not on GPU, but on CPU or memory
  说明即使将不同模型运行在不同GPU上，依然无法获得加速，猜想应该不是GPU上的问题，应该是CPU或者内存
- 通过查询资料，在[here](https://github.com/keras-team/keras/issues/9204#issuecomment-370805961) 得到启发，可能是因为数据增强增加了CPU占用导致无法获得加速

- Result on 
  - Model: mobilenet_v2 
  - Input size: 224x224 
  - GPU: 4 Titan Xp 
  - Dataset: dataset-v11
    
    |Batch Size|GPU number  |sing gpu mem|CPU util v7|CPU util v11|Epo tm v7 |Epo tm v11|
    |:--------:|:----------:|:----------:|:---------:|:----------:|:--------:|:--------:|
    |  16      |1           |2361M       |58.0%      |19.5%       |18.10s    |16.5s     |
    |  32      |1           |4175M       |59.0%      |20.5%       |16.20s    |14.4s     |
    |  64      |1           |7627M       |61.0%      |20.5%       |15.40s    |13.6s     |
    | 128      |2           |7737M       |76.6%      |25.0%       |11.30s    | 8.7s     |
- As shown in the table above, after the dataset was changed from V7 to V11, although the "video memory" remained unchanged, the "CPU util" and "epoch time" were significantly reduced, because the crop and resize operations were removed in the transform step of the Dataset-V11
- As I guessed, when the three mobilenet_v2 "batch sizes "of 32, 64, 128 were trained simultaneously, the "epoch time" did not appear as it did when running a model
In the previous situation, when the overall "CPU util" was only 55% according to the monitoring, `Parallel` was really implemented. Before, because the three models were running at the same time, the maximum capacity of CPU was already exceeded.
So there is no real `Parallel`, only `Concurrency`. That is why there was a situation where three models were running at the same time, and the training time of a single model was significantly increased
- After the above changes, mobilenet_v2 can still get the same top-1 as before, which proves that this is a cost-free trick
#### 如上表所示，将数据集从v7换成v11之后，虽然显存占用量不变，但是，CPU使用量大幅降低，"epoch time"也有所降低，因为在dataset-v11的transform步骤中去掉了crop和resize的操作
如我猜想的一样，当按照如上配置的"batch size"分别为32 64 128的三个mobilenet_v2同时训练时，"epoch time"与运行一个模型时一样，并没有出现
之前的那种大幅增加的情况，而通过监控可以看到总体的CPU使用量也只有55%，此时才真正实现了Parallel，之前因为三个模型同时运行时，已经超过了CPU的最大承受能力，
所以无法实现真正的Parallel，只能实现Concurrency，这也就是为什么之前会出现三个模型同时运行时，单个模型的训练时间比之前大幅增加的情况
在以上改动之后mobilenet_v2依然可以获得之前一样的Top-1，证明这是一种cost-free的提升

### Feb 07, 2021
- Result on 
  - model: resnet34
  - input size: 224x224
  - GPU: 4 Titan Xp
  - Torch: 1.5.1
- Notes: 
  - Video memory: the maximum video memory on a single gpu
    
    |Batch Size|Video memory|Growth rate|Epoch time|Growth rate|
    |:--------:|:----------:|:---------:|:--------:|:---------:|
    |  16      |1259M       |None       |29.10s    |None       |
    |  32      |1539M       |22%        |16.70s    |-43%       |
    |  64      |2047M       |33%        |12.30s    |-26%       |
    | 128      |3233M       |58%        |11.40s    |- 7%       |
    | 256      |5527M       |71%        |11.40s    |  0%       |
    | 512      |9905M       |79%        |12.80s    | 12%       |
- The above table shows that with the increase of the batch size, the "epoch time" will not always decrease, and may even increase, that is to say, increasing the "batch size" does not necessarily accelerate the training.
- On the contrary, "video memory" usage increases all the time, and the growth rate is increasing.Therefore, the method of blindly increasing "batch size", apart from its influence on the learning effect, is not cost-effective in terms of efficiency.
- Only when the "epoch time" decreases faster than the "video memory" growth rate, the efficiency will be improved

上表说明：随着batch size的增大，Epoch time并不会始终随之减小，甚至还可能变大，也就是说增大batch size并不一定能够加速训练，
反而是显存占用量，始终随之增大，且增速越来越大。所以一味增大batch size这种方法，抛开其对于学习效果的影响不说，首先其在效率上也是不划算的，
只有当epoch time的降低速度大于显存的增长速度时，才将获得效率上的提升

