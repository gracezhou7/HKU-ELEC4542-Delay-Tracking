数据集与划分
1) DanceTrack
用官方的40个Train当作我们的Train
官方的25个Val当作我们的Val和Test（10val&15test），因为官方没有给Test的GT
FPS：统一30


2) MOT17
公开7段Train分为Train：Val：Test = 5：1：1
FPS：以各序列seqinfo.ini中的frameRate为准（不同序列帧率不同；例如官方MOT17-05为14 FPS）


3) DTB70
70段分为Train：Val：Test = 50：10：10
FPS：统一30

Baseline模型家族
1) No Compensation 
2) Linear Extrapolation
3) CV-Kalman：其中Q需要训练
4) CA-Kalman：Q需要训练
5) IMM(CV+CA)：Q需要训练
6) Polynomial
7) PVT++P_M(复现)：这个模型需要训练，和LSTM以及Transformer用同一套训练标准。


学习方法（深度学习对照组）
1）LSTM
主要结构：2层LSTM（hidden128）+延迟嵌入拼接到每个输入步，输出4维相对运动
2）Transformer
主要结构：6层encoder，d=128，4 head；token=[m_i; Δf_i; e(Δ) ]


实验步骤/代码实现
1）Step1 数据预处理与加载
step1.1 读取DanceTrack, MOT17, DTB70的原始GT标注，过滤掉丢失或无效的帧（包括目标被完全遮挡或移出画面）
step1.2 Sliding Window滑动窗口切片：对每个视频序列，按固定步长滑动，通过截取过去连续k帧（如k=8）的目标框，生成观测集O。在线计算归一化位移 mi、时间间隔 fi、以及相对运动因子 p。
step1.3 在训练时对每个Episode随机采样不同的延迟Δ（例如从33ms到200ms之间均匀采样），并生成对应的傅立叶延迟编码e(Δ）

2）Step2 Training and Hyperparameter Tuning
（这一步主要是针对我们在这个项目里提出的基于学习的预测方法）
step2.1 训练集训练（Train）：使用Smooth L1 Loss，将归一化位移输入网络，监督输出预测的相对运动m_hat。使用相同的优化器（如AdamW）进行梯度下降，使用相同的训练轮数（Epoch）。
step2.2 验证集调参（Validation）
给LSTM和Transformer相同的最大搜索次数（Hyperparameter包括Weight Decay，Learning Rate，Dropout Rate）。所有超参的好坏，都只能根据Val集上的平均误差来评判。

3）Step3 Evaluation
当所有模型（前面提到的所有Baseline方法，以及我们训练的LSTM和Transformer）都准备好后，进入最后的统一对决和评估。使用前面对3个数据集划分好的Test集合，让所有模型在相同的Episode、相同的观测集O、相同的延迟Δ下进行测试。传统方法直接使用调好q的公式计算。深度学习方法吃入历史特征e（Δ）吐出预测框b_hat (t)。
评判指标分为两个：ADE（Average Displacement Error) 和 FDE（Final Displacement Error)。ADE是整条预测轨迹与真值的平均欧式距离。FDE是最终目标时刻t的断点误差。


4）整理结果&分析
step4.1 画出延迟曲线：以延迟Δ为横轴，误差（ADE）为纵轴，画出各条方法的性能衰减曲线。
step4.2 寻找临界点：用Bootstrap 95% 置信区间配对检验，找出学习方法超越传统KF的*（Crossover Point）
step4.3计算DanceTrack各序列的法向加速度a，验证*2/a  的定量预言。



备注
实验里的所有延迟都是通过代码人为假设的、采样和注入的，而不是连着真机在实施卡顿中测出来的
项目仓库：https://github.com/gracezhou7/HKU-ELEC4542-Delay-Tracking.git
PVT++里提出的baseline方法虽然页用到了神经网络，但架构很轻量很简单。他的Motion Branch用的是1D Conv来提取多帧之间的时序特征。Visual Branch用的是3D Conv来处理时空图像特征。在我们的项目中，我们引入了更强大的时序建模架构：LSTM是用RNN去显式捕捉长距离的时间前后依赖关系；Transformer是用 6 层 Encoder 和Multi-Head Attention机制。Transformer 在处理长序列、复杂非线性运动（如 DanceTrack 里杂乱的跳舞轨迹）时，表达能力和全局感受野远超传统的 1D/3D 卷积
e()是怎么来的：训练时，代码写一个随机抽样函数（random.uniform()。每次随机抽取一个毫秒数，比如抽到66代表模拟系统卡顿66ms，也就是在告诉模型去预测66ms之后的位置。


代码结构与数据准备
------------------

项目的数据加载代码目前放在仓库根目录的 `data/` Python 包中；原始数据不会提交到 Git。将下载并解压后的数据放在 `data/raw/`，按以下目录组织：

```text
data/raw/
├── DanceTrack/
│   ├── train/<sequence>/gt/gt.txt
│   └── val/<sequence>/gt/gt.txt
├── MOT17/ 或 MOT17Labels/
│   └── train/<sequence>-<detector>/{gt/gt.txt,seqinfo.ini}
└── DTB70/
    └── <sequence>/{groundtruth_rect.txt,img/}
```

当前数据加载器读取 MOTChallenge 的 `gt/gt.txt` 和 DTB70 的 OTB `groundtruth_rect.txt` 两种格式。DanceTrack 与 DTB70 使用项目设定的 30 FPS；MOT17 从每段序列的 `seqinfo.ini` 读取真实帧率。注意官方 MOT17-05 为 14 FPS，因此 MOT17 实际帧率并非只有 25/30，延迟换算必须逐序列进行。MOT17 若提供多个检测器版本，只使用 FRCNN 版本（没有时按名称确定性选择一个），避免同一段视频跨划分或重复计数。只使用 GT 框的模型不需要图像；`MOT17Labels/train` 已包含 `gt/gt.txt`，可直接用于当前管线。

序列划分使用固定随机种子 `4542`：DanceTrack 官方 train 全部用于训练，官方 val 按 10/15 划为 val/test；MOT17 的 7 段按 5/1/1 划分；DTB70 的 70 段按 50/10/10 划分。数据目录序列数不符合预期时会报错，不会静默使用不完整数据。仅调试部分数据时可传 `strict=False` 或使用下面的 `--allow-missing`。

### 数据预处理流程

1. **读取与过滤**：只保留有效行人框，剔除置信度或可见度为 0、非行人类别、宽高非正、完全出画的框。MOT17 的 `gt.txt` 中类别 2/4/7/8/9 会被过滤，只保留类别 1。
2. **滑窗**：以每个 track 连续 `history=8` 帧为观测窗口，锚点是最后一个观测帧；按 `stride=8` 帧取下一个窗口。窗口到最大延迟对应的未来帧必须持续可见，否则不生成样本。
3. **特征**（`data/features.py`）：
   - 框为像素坐标 `xywh`，中心点 `c=(x+w/2, y+h/2)`。
   - 归一化位移：`m_i = [(cx_i-cx_{i-1})/w_{i-1}, (cy_i-cy_{i-1})/h_{i-1}]`，用上一个观测框宽高归一化。
   - 时间间隔：`Δf_i=(t_i-t_{i-1})`，单位为秒。
   - 相对运动因子：`p_i = ||m_i|| / Δf_i`，即每秒归一化位移（归一化速度）。
   - 4 维监督：`[(cx_t-cx_k)/w_k, (cy_t-cy_k)/h_k, ln(w_t/w_k), ln(h_t/h_k)]`。
   - 延迟编码：`e(Δ)=[sin(2π·2^j·Δ/Δmax), cos(2π·2^j·Δ/Δmax)]`，`j=0..7`，共 16 维。
4. **模拟延迟**：训练时从 33–200 ms 均匀采样 `Δ`；验证/测试固定延迟。目标框在 `anchor_time + Δ` 处对相邻 GT 帧做线性插值，而不是四舍五入到帧，避免 14 FPS 序列把 33 ms 量化成约 71 ms。
5. **一致性**：验证/测试的所有延迟共用同一批锚点（按最大延迟 200 ms 建索引），保证不同 `Δ` 下样本完全配对。

运行预处理检查（校验标注、统计窗口；只处理已存在的数据集）：

```bash
.venv/bin/python -m data.preprocess --root data/raw --allow-missing
```

默认结果写入 `data/processed/preprocess_summary.json`，其中包含每个数据集/划分的序列数、track 数、框数、窗口数、FPS 和延迟对应未来帧数。当前只有 `MOT17Labels` 时也能运行；补齐 DanceTrack 和 DTB70 后去掉 `--allow-missing` 即可严格校验 40/25、7、70 段。

项目代码可在 macOS 和 Ubuntu 上运行，两个环境各自使用仓库内的 `.venv` 虚拟环境。首次只需在每台电脑分别执行一次安装命令：

```bash
./scripts/setup.sh
```

脚本会检查 Python 3.10+、创建 `.venv`、安装 PyTorch 和本项目（editable 模式）、运行测试，并检查可用的 GPU 后端。macOS 安装默认 PyTorch，可使用 CPU 或 Apple Silicon 的 MPS；Ubuntu 默认安装 CUDA 12.8 版 PyTorch，供 NVIDIA GPU 使用。训练仅计划在 Ubuntu + NVIDIA GPU 上运行；macOS 用于开发、数据准备和测试，不要求在其上训练。Ubuntu 若缺少 `venv` 模块，先执行 `sudo apt install python3-venv`；如默认 Python 版本不合要求，可安装 Python 3.10+ 并通过 `PYTHON=python3.10 ./scripts/setup.sh` 指定。若 Ubuntu 的 NVIDIA 驱动与默认 CUDA wheel 不兼容，可设置 `TORCH_INDEX_URL` 为兼容的 PyTorch CUDA wheel index 后重跑安装脚本。

进入项目环境：

```bash
source .venv/bin/activate
```

之后运行 Python、测试和训练命令时都使用此虚拟环境中的解释器；运行 `.venv/bin/python -m unittest discover -s tests -v` 可再次执行测试。Ubuntu 安装完成后若 CUDA 不可用，请先确认 NVIDIA 驱动正常（`nvidia-smi`），再检查 PyTorch CUDA 运行时。

快速创建 PyTorch DataLoader：

```python
from data import build_data_loaders

loaders = build_data_loaders(
    root="data/raw",
    batch_size=128,
    num_workers=4,
    history=8,
    stride=8,
    delay_range_ms=(33, 200),
    eval_delay_ms=100,
)
batch = next(iter(loaders["train"]))
```

batch 提供 `history_boxes`（像素 xywh）、`normalized_motion`（7×2）、`time_intervals`（7）、`relative_motion_factor`（7）、`delay_encoding`（16）、`target_box`（4）、`target_motion`（4），以及 `frame_ids`、`anchor_frame`、`target_frame_position`、`target_time_ms`、`delay_ms`、`fps`、`sequence` 和 `track_id`。训练集每个样本随机采样延迟；val/test 使用固定延迟。

延迟曲线评估可用 `build_eval_loaders`，它会让每个延迟共用同一批锚点：

```python
from data import build_eval_loaders

eval_loaders = build_eval_loaders(
    root="data/raw",
    delays_ms=(33, 66, 100, 133, 166, 200),
    split="test",
    batch_size=128,
)
for delay_ms, loader in eval_loaders.items():
    batch = next(iter(loader))
```

检查 CUDA 是否可用：

```bash
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable")'
```
