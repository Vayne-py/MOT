MOT
===
## FairMOT：A Simple Baseline for Multi-Object Tracking
### Accomplished the Object detection and re-identification tasks in a single network.

### Tracking performance
Dataset|MOTA|IDF1|IDS|MT|ML|FPS
:----:|:----:|:----:|:----:|:----:|:----:|:----:|
2DMOT15|60.6|64.7|591|47.6%|11.0%|30.5
MOT16|74.9|72.8|1074|44.7%|15.9%|25.9
MOT17|73.7|72.3|3303|43.2%|17.3%|25.9
MOT20|61.8|67.3|5243|68.8%|7.6%|13.2

[项目地址](https://github.com/ifzhang/FairMOT)

[Demo，提取码：1111](https://pan.baidu.com/s/1yT-q5z7ljlX4RHoFpMf3Ng)

[FairMOT_XMind.PDF](https://github.com/Vayne-py/Images/raw/main/FairMOT_XMind.pdf)
#### 网络结构
![Aaron Swartz](https://github.com/Vayne-py/Images/raw/main/FairMOT_Network.png)

### 整体代码框架

代码块|.py file|具体功能
:----:|:----:|:----:|
配置文件|opts.py|设置数据集、head、loss、及相关路径等参数
数据读取|jde.py|数据读取，绘制hm等
Network|pose_dla_dcn.py|编码解码网络，提取高分辨率特征图
Loss|mot.py|定义各heads结构的loss
Track|multitracker.py|沿用deepsort跟踪算法进行多目标跟踪
Detector|CenterNet中detection部分|检测目标，给跟踪网络提供哥目标的位置大小信息

### 具体跟踪机制
##### FairMOT跟踪机制沿用Deepsort跟踪机制(multitracker.py)
###### 变量说明
    状态变量：
    activated：激活状态，用于单次误捡目标的判断
    track_state：跟踪状态，共有tracked、lost、remove三种
    容器：
    unconfirmed_stracks(activated = F, track_state=tracked ) 只出现一次的目标（检测器误检的目标）
    activated_stracks(activate=T, track_state=tracked) 跟踪状态良好的tracker
    lost_stracks(activate=T, track_state=lost)激活，但是跟踪状态丢失
    refind_stracks（activated=T, track_state=lost->tracked）跟丢之后重新找回的目标
#### 具体步骤如下：
###### 第一步：获得当前帧(第一帧)的检测框和id特征
    1.用CenterNet检测得到目标的(x,y,w,h)和embeddings的id特征，根据设置的置信度conf_thres筛选检测框和其对应的id特征。
    2.对每个筛选出的目标初始化一个tracker，n个目标共有n个tracker。
    3.将n个tracker放入activared_stracks容器中。
###### 第二步：首次将当前帧(第二帧)与前一帧进行外观+距离匹配
    1.将激活的tracked_stracks和lost_stracks融合成strack_pool
    2.对strack_pool进行卡尔曼滤波，更新内部的均值与方差。
    3.计算Strack_pool与detections之间的embedding距离(cosine距离)，对比的是当前帧的id特征和之前帧的线性累积特征(smooth_feat)，获得特征对比的cost_matrix。
    4.融合运动特征，计算卡尔曼滤波与当前检测直接的距离，如果距离过大则将cost_matrix中的距离设置为无限大(考虑不存在太大的位移情况)，再将卡尔曼滤波预测的结果和特征的结果做成距离加权，得到考虑了运动状态的cost_matrix。
    5.执行匈牙利算法获得匹配结果，得到未匹配的track和detection。
    6.将匹配成功的detection根据之前帧的跟踪状态(是否处于跟踪)加入activared_stracks和refind_stracks中，更新frame_id、卡尔曼滤波的均值方差、置信度分数和id特征。
###### 第三步：第二次将当前帧与前一帧进行匹配，通过IOU
    1.将第二步中没有匹配成功的track和detection计算IOU距离。
    2.执行第二步中的(5)(6)步。
    3.将再不匹配的track修改为lost状态。
    4.与不确定序列做匹配，计算IOU距离，并执行第二步中的(5)(6)步。
    5.如果不确定序列没有获得匹配，则将其加入到removed_stracks中。
###### 第四步：初始化新序列
    判断依然没有匹配的detection的置信度，高于conf_thres则初始化一个新序列，低于则丢弃。
###### 第五步：更新状态
    1.将处于丢失状态超过max_time_lost(30)的序列计入删除序列中。
    2.更新self.tracked_stracks，将跟踪状态为activared_stracks和refind_stracks的序列合并。
    3.将新增的lost_stracks加入self.lost_stracks中，并去除重新激活跟踪和删除的序列。
    4.更新remove_strack。
    5.删除lost与activared_stracks中的重复序列。
###### 第六步：输出
    输出确定的跟踪
### tensor流动
    tensor流动主要是在数据预处理及detection模块，跟踪部分并未使用深度学习方法。
#### 一. 数据读取及预处理(jde.py)
    首先是读取数据并根据高斯分布来对中心点进行处理(应用高斯分布而不将中心点标为1，其他点标为0是由于中心点周围的点同样可以很好地描述行人)。
    default_resolution = [1088, 608]
#### 二. detection部分(CenterNet)
##### decode.py(将hm解码成b-box)
###### 1.经过_nms()：    
    def _nms(heat, kernel=3):
        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep
    寻找8邻近极大值带点，keep为h极大值点的位置，返回heat*keep，筛选出极大值点，为原值，其余为0。
###### 2.经过_topk():
    def _topk(scores, K=40):
      batch, cat, height, width = scores.size()
      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K) #topk_scores, topk_inds:[batch*cat*k] index[0,W×H-1],每张heatmap（每个类别）中前K个最大的score和id
      topk_inds = topk_inds % (height * width)
      topk_ys   = (topk_inds / width).int().float()
      topk_xs   = (topk_inds % width).int().float()
      topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K) #topk_score, topk_ind:[batch*K] index[0,cat×k-1]
      topk_clses = (topk_ind / K).int()
      topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)  #return feat:batch*K*1 index[0,cat×k-1]
      topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
      topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
      return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
      返回四个值：
                topk_score:batch*k,每张图片中最大的k个值
                topk_inds:batch*k，每张图片中最大的k个值对应的index，index∈[0,WxH-1]
                topk_ys, topk_xs:batch*k，每张图片中最大的k个值的横纵坐标
###### 3.经过_tranpose_and_gather_feat():
      将_topk得到的index用于取值，函数的输入有reg和wh，reg用于回归，wh计算得到bbox的WH
      def _tranpose_and_gather_feat(feat, ind):           #feat:[batch*C*W*H],ind:[batch*k]
        feat = feat.permute(0, 2, 3, 1).contiguous()      #C放到最后并将内存变为连续，用于view
        feat = feat.view(feat.size(0), -1, feat.size(3))  #feat"[batch*(W*H)*C]
        feat = _gather_feat(feat, ind)
        return feat
###### 4.经过mot_decode(),获得每个筛选出的目标的[x,y,w,h],用于跟踪
