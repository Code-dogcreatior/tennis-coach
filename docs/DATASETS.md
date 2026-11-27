# 数据来源核验清单

核验日期：2026-11-17。“未说明许可”≠“允许商用”。下表只记录**实际访问核验过**的情况；未核验的写明未核验。

## 1. 产品主路径：自有/教练示范视频

* 通过 Web 界面「参考库 → 上传示范视频」或 `POST /api/videos (kind=reference)` 导入，**不依赖任何公开数据**。
* 每个视频保存：来源、许可/授权说明、是否允许商用、是否允许再分发、球员、持拍手、机位说明、动作类型、帧率/时间尺度（慢动作倍率）、审核状态（参考示范 / 教练审核+审核人）。
* 所有修改写入纠错日志（`corrections` 表，`GET /api/videos/{id}/corrections`）。

## 2. 公开数据集

| 数据集 | 官方来源 | 覆盖动作 | 视角 | 标注内容 | 下载方式 / 实测状态 | 许可 | 商用 | 再分发 | 本项目使用 |
|---|---|---|---|---|---|---|---|---|---|
| **THETIS** | http://thetis.image.ece.ntua.gr/ → 重定向到 https://github.com/THETIS-dataset/dataset | 12 类：双反、单反、反手切削、反手截击、正手平击、开放式正手、正手切削、正手截击、平击/上旋/切削发球、高压 | **正面**（Kinect，室内，无球空挥），640×480，18 fps | 每段一个动作类别；55 名球员（p1–p31 初学者，p32–p55 熟练者）；另有深度/轮廓/Kinect 骨架 | GitHub 原始文件可直接下载（无需登录），RGB 每类约 320 MB；**已实测下载** 7 类 × 22 名球员子集 | **仓库未声明许可（未说明）** | 不可假定 | 不可假定 | 仅本地评测：动作识别/拒识的按球员隔离测试。不再分发、不打包进产品 |
| **CalTennis** | https://huggingface.co/datasets/demalenk/caltennis ，论文 arXiv:2606.20542 | 实战/练习中的网球动作（未见挥拍类别标注） | 多机位（2–6 台 iPhone，60 Hz） | 多视角同步视频+时间戳，用于单目→3D 姿态评测 | Hugging Face 公开、未设门槛（API 核验：gated=False）；**未下载**（体量 ~51 小时） | **CC BY-NC 4.0** | **否** | 可（署名、非商用） | 首版未用。后续可用于多机位一致性/3D 方案评估（仅非商用研究） |
| **US Open serve dataset** | https://github.com/jasnwag/tennis_serve_dataset | 仅发球（5,966 次，109 名职业球员） | 转播视角（底线后方） | RTMPose 2D → MotionBERT **3D 姿态序列**、关节角度、发球类型/速度、比赛上下文；不含原视频 | GitHub 可访问（仓库元数据已核验）；**未下载数据** | LICENSE 文件为 **CC BY 4.0**（GitHub API 显示 NOASSERTION，因非标准识别） | 是（署名） | 是（署名） | 首版未用：数据是 3D 升维序列，与本项目“同机位 2D 比对”不直接兼容；可作为后续发球 3D 参考来源 |
| **DynSports** | 未找到 | — | — | — | 检索未找到该名称的数据集/论文，**无法核验** | — | — | — | 未用 |
| Tennis Player Actions Dataset (Mendeley) | https://data.mendeley.com/datasets/nv3rpsxhhk | 正手/反手/准备/发球 | 自拍 | 单帧图像 + COCO 关键点（每类 500 张） | Mendeley 公开 | CC BY 4.0（API 核验） | 是 | 是 | 未用：单帧图像，没有时序，无法用于挥拍分析 |

## 3. 本地测试视频（YouTube）

`scripts/fetch_youtube.ps1` 记录了下载命令；每个视频旁保存 `*.info.json`（URL、上传者、许可字段）。

| 视频 ID | 内容 | YouTube 许可字段 | 用途 |
|---|---|---|---|
| 5lf-t4obkkU | Fonseca 侧/斜后方正手+双反练习（多机位剪辑，4K60 下载 1080p） | CC BY | 多人干扰、背面/斜后方、剪辑场景的失败案例 |
| nW5QJN-0EeQ | 业余球员正面正手 | CC BY | 演示参考库（正手，正面） |
| Rw3ZDfTTD9M | Nadal 正面练习（左手） | CC BY | 左手/镜像演示 |
| pg_dxIvJRZo, WsQ0ghZg5FE, eBa-sy2Mt1s, kJUuzlk6d8M, rTXgMbyjMuU, gdJXQ69yuMw, TdlXwYS_Neo, 5_SnJPl4cz4 | 职业球员发球/反手慢动作合集 | **标准 YouTube 许可（非 CC）** | 仅本机测试，不入参考库、不再分发；用户已同意本地测试使用 |

另有 5 个侧面视频（含 Wawrinka 侧面单反）因 YouTube 限流（HTTP 429 / 需登录）未能下载；未使用 cookies 绕过。

## 4. 模型权重许可

| 组件 | 来源 | 许可 | 备注 |
|---|---|---|---|
| rtmlib（推理封装） | github.com/Tau-J/rtmlib | Apache-2.0 | |
| RTMPose-m body7 256×192 ONNX | OpenMMLab download.openmmlab.com | 代码/权重随 MMPose 发布，Apache-2.0 | 训练数据 body7 = COCO、AI Challenger、CrowdPose、MPII、sub-JHMDB、Halpe、PoseTrack18；其中部分数据集自身有非商用条款（如 AI Challenger、MPII），**商用前需法务确认** |
| YOLOX-m humanart ONNX | OpenMMLab | Apache-2.0 | 训练数据含 Human-Art（研究用途许可），商用前需确认 |
| OpenVINO / ONNX Runtime | Intel / Microsoft | Apache-2.0 / MIT | |
| MediaPipe Pose Landmarker（仅基准对比，未进主流程） | Google | Apache-2.0 | |

## 5. 数据缺口与补拍清单

见 [RESHOOT.md](RESHOOT.md)。
