# 网球视频动作训练助手（原型）

从视频到训练反馈的本地闭环：**示范入库 → 用户上传 → 姿态提取 → 质量检查 → 持拍手/机位识别 → 动作识别（正手/单反/双反/发球/未知）→ 阶段切分 → 参考匹配（分阶段 DTW）→ 指标对比 → 带证据的反馈**。所有视频都在本机处理，不上传到第三方。

> 这是诚实的原型：识别和反馈都基于 2D 姿态，附带置信度和证据，不能识别或证据不足时直接说明。真实评测结果见 [docs/REPORT.md](docs/REPORT.md)，已知局限见报告第 7 节。

## 快速开始（Windows，已在 i5-1235U + Iris Xe 上验证）

前置：[uv](https://docs.astral.sh/uv/)、ffmpeg（需要在 PATH 中）。无需 NVIDIA 显卡：姿态模型通过 OpenVINO 在 Intel 核显上运行，核显不可用时自动回退 CPU。

```bash
uv sync
```

```bash
uv run tennis-coach
```

浏览器打开 http://127.0.0.1:8765 。

模型权重已随仓库提供（Git LFS，`models/`）：RTMPose-m（body7，256×192，54 MB）和 YOLOX-m（HumanArt，101 MB）。克隆前请先安装 [Git LFS](https://git-lfs.com/)（`git lfs install`）；如果克隆时没有拉到 LFS 文件，程序会自动改为从 OpenMMLab 下载同一份权重。

仓库不包含任何视频、姿态缓存和数据库（`data/` 下只提交 `labels/` 标注）。克隆后参考库是空的，导入**职业参考包**就能直接做职业对比：

```bash
uv run python scripts/ref_pack.py import
```

参考包 `refpack/pro_reference_pack.zip`（60 MB，Git LFS）包含 20 段职业训练视频里 356 条参考挥拍（发球 198、正手 70、单反 44、双反 44）的派生数据：2D 关键点序列、特征、指标、剪辑点和标注。**没有任何视频画面。** 导入后：

* 动作识别、分阶段 DTW 匹配、职业检查点范围、“主要问题”和练习建议都和完整参考库一致。用 19 段演示视频对比过：最接近的职业挥拍和报出的问题完全相同。
* 关键帧对照图和同步播放里，职业一侧显示为骨架，标有“参考包：只有骨架，没有原视频”。
* 参考包条目没有视频也可以重新分析，直接用包里的关键点。

如果想看职业球员的真实画面，可以先 `ref_pack.py remove`，再用 `fetch_pro.py` 下载原视频、`build_pro_library.py` 在本地重建（需要几个小时的姿态提取）。

**注意**：参考包里只有职业参考。THETIS 业余参考的许可不允许再分发，所以没有放进来，这会影响业余视频的动作识别。只用参考包时，THETIS 上被拒识为“未知”的目标动作从 19% 升到 29%，平击发球 66 段里识别出 42 段（完整参考库是 54 段）；识别出来的动作正确率不变（97%）。上传几段自己的示范视频到参考库可以缓解。

> 已知环境问题（2026-11）：Windows 更新后，如果 `C:\Windows\System32\msvcp140.dll` 版本低于 14.40，`import onnxruntime` 会直接崩溃（rtmlib 导入时会加载 onnxruntime）。安装最新的 VC++ 运行库即可修复：`winget install Microsoft.VCRedist.2015+.x64`。

使用流程：

1. 「参考库」→ 上传示范视频（建议声明动作类型、持拍手、是否慢动作、来源与授权）。系统会自动切分每一次完整挥拍，生成参考条目，可以修改标签、审核状态和风格，也可以停用或删除。
2. 「分析我的视频」→ 上传待分析视频 → 查看处理进度 → 查看结果：**与职业动作对比**（主要问题、你和最接近的职业挥拍的关键帧对照、数值与职业范围、练习建议）、骨架叠加、与参考同步播放（按 DTW 对齐或按击球点对齐）、阶段时间线、指标曲线、指标对比、反馈（每条都附证据时间段）、质量检查、识别依据和置信度。
3. 识别有误时，在「人工纠错」中修改持拍手/动作/机位/时间尺度。修改会记录在纠错日志里，并自动重新分析（姿态已缓存，几秒即可完成）。
4. 「删除视频及派生文件」会删除原视频、转码文件、姿态缓存、分析结果，以及由它生成的参考条目。

## 发球部分：目前的测试情况

发球机位相对固定，是目前优化最多、验证最充分的动作。完整数据见 [docs/REPORT.md](docs/REPORT.md) 第 8b–8e 节。所有结论都基于 2D 姿态，没有球和球拍检测，也没有经过教练审核。

**职业发球参考库**：198 条，来自 10 组职业球员的公开训练视频。按机位分：背面 107、正面 59、侧面 32。7 段慢动作视频被排除（转场太多，系统判定的“击球”帧经常还停在奖杯姿势）。保留的发球逐条看过关键帧。

**技术检查点**：门槛是职业留一球员误报 ≤ 15%、样本 ≥ 10，并且 THETIS 上新手的触发率不能明显低于有经验球员。发球检查点只和同机位的职业样本比较。

| 检查点 | THETIS 新手触发 | 有经验球员触发 | 职业误报 | 状态 |
|---|---|---|---|---|
| 抛球手没有充分上举 | 64% | 24% | 0.5% | 可靠 |
| 击球点偏低 | 46% | 21% | 6% | 可靠 |
| 击球时手臂没有伸直 | 24% | 6% | 4% | 可靠 |
| 奖杯姿势肩部没有倾斜 | 97% | 73% | 9% | 通过，但过于敏感 |
| 击球时抛球手没有回收 | 9% | 18% | 10% | 通过，区分度存疑 |
| 蹬伸不足 | 61% | 52% | 17% | 未通过，只作参考提示 |
| 持拍手下落不充分 | 24% | 6% | 19% | 未通过，只作参考提示 |
| 奖杯姿势屈膝不足（侧面） | — | — | 45% | 未通过 |

职业发球按留一验证，23% 会被报出至少一条主要问题，这是目前的误报水平。

**发球识别**（`scripts/eval_knn_library.py`，完整参考库 = 职业 + THETIS 演示参考，类别均衡投票；只导入参考包时的差别见上文“快速开始”）：

* 职业发球（留一球员）：98.5% 被识别为发球。
* THETIS 平击发球 66 段：54 段识别为发球，12 段拒识为“未知”，没有被判成其他动作。
* **高压和发球在 2D 骨架上很难区分**：THETIS 66 段高压里，38 段仍被判为发球。系统用“抛球前非持拍手有没有降到髋部以下”区分，看起来像高压时会显示“可能是高压”并把置信度压到 35%：被判为发球的高压中有 17/38 被标记，真发球只有 4/54 被误标。这个特征在训练组和测试组上的表现不稳定，直接拒识会丢掉约 1/5 的业余发球，所以只做提示，不做拒识。

**真实场地（已标注的 1 段斜前方业余发球视频，8 次发球）**：8 次全部检出并判为发球，击球时刻中位误差 −33 ms（早 1 帧），±3 帧内 100%。这段视频里球员弯腰捡球的 3 个动作原来被判成正手/单反，现在按“手腕降到膝盖以下”判为非挥拍（见报告 8g）。

**尚未覆盖**：没有真实场地的侧面业余发球标注数据（侧面检查点没有在业余球员上验证）；左手球员的职业发球很少；还没有教练对诊断结果做过复核。

## 命令行与脚本

| 命令 | 作用 |
|---|---|
| `uv run pytest -q` | 单元测试，共 34 个（镜像标签语义、左右手等价、阶段顺序、节奏保留、DTW、可变帧率、左右跳变修正、跟踪重选、同机位职业范围、kNN 投票方式、高压提示、弯腰捡球拒识、标注接口安全、YouTube 链接校验、提示时间换算、按标注人分文件） |
| `uv run python scripts/inspect_video.py <video> [--hand left] [--ts unknown] [--sheet out.jpg]` | 命令行分析单个视频，输出关键事件帧拼图 |
| `uv run python scripts/fetch_thetis.py` | 下载 THETIS RGB 子集（可复现，生成 manifest，带许可说明） |
| `uv run python scripts/thetis_apparent_hand.py` | 生成 THETIS 每名球员“画面中持拍手”的真值（仅用于评测） |
| `uv run python scripts/eval_thetis.py --hand gt` / `--hand auto` | 按球员隔离的评测，输出到 `reports/` |
| `uv run python scripts/eval_knn_library.py` | 用应用实际参考库比较 kNN 投票方式（plain / source / balanced）在 THETIS 和职业留一上的识别效果 → `reports/knn_library_eval.json` |
| `uv run python scripts/fetch_eval_real.py` | 下载真实场景评测片段（8 段 CC BY 业余视频，清单 `data/labels/eval_real_manifest.csv`），并转换成标注要求的格式 |
| `uv run python scripts/prep_annotate.py <视频…>` | 批量把自己的视频转换成恒定帧率、首帧 0 秒、无音频的 MP4，放进 `data/raw/annotate/<名字>/` 供标注（单个视频也可以直接在标注页“导入视频”） |
| `uv run python scripts/eval_real.py [--annotator 名字]` | 用 `/annotate.html` 标注的结果评测检出率、动作识别、击球时刻误差和持拍手 → `reports/eval_real.md`（标注说明见 [docs/ANNOTATION.md](docs/ANNOTATION.md)） |
| `uv run python scripts/ref_pack.py export / import / remove` | 导出、导入、删除职业参考包（只有关键点等派生数据，没有视频画面），默认文件 `refpack/pro_reference_pack.zip` |
| `uv run python scripts/fetch_pro.py [--only ID ...]` | 按 `data/labels/pro_manifest.csv` 用 yt-dlp 下载职业训练视频到 `data/raw/pro/<id>/`（公开视频、不登录；`section` 列可只截一段） |
| `uv run python scripts/build_pro_library.py` | 把 `data/labels/pro_manifest.csv` 中的职业训练视频（先用 yt-dlp 下载到 `data/raw/pro/<id>/`）作为职业参考导入；`pro_autolabel.py` 生成挥拍标签后，加 `--review-only` 应用标签 |
| `uv run python scripts/eval_pro_compare.py` | 职业对比的验证：职业球员留一误报率 + THETIS 新手/有经验球员触发率；生成运行时使用的检查点验证表 `src/tennis_coach/checkpoint_validation.json` |
| `uv run python scripts/reprocess_offline.py --pro-done --queries` | 在本进程内重新分析已入库视频（姿态已缓存） |
| `uv run python scripts/run_demo.py` | 通过 API 跑完整的真实闭环演示（示范入库 → 上传 → 结果），输出到 `reports/demo/` |
| `uv run python scripts/bench_pose.py <video>` | 姿态后端测速 |
| `powershell -File scripts/fetch_youtube.ps1` | 下载本地测试用 YouTube 片段（只用公开视频，不使用 cookies） |

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/videos` | 上传（multipart：`file`、`kind`=reference/query、`meta`=JSON：hand、action、time_scale、player、source、license、license_commercial、redistribution、review_status、reviewer、camera、notes） |
| GET | `/api/videos?kind=` | 列表（状态、进度、摘要） |
| GET | `/api/videos/{id}` | 详情 + 完整分析结果 |
| GET | `/api/videos/{id}/status` | 任务状态、阶段、进度、失败原因 |
| PATCH | `/api/videos/{id}` | 元数据/纠错（hand、view_override、time_scale、action、action_overrides…），涉及分析的字段会自动重新排队 |
| POST | `/api/videos/{id}/reanalyse` | 重新分析 |
| DELETE | `/api/videos/{id}` | 删除视频及全部派生文件 |
| GET | `/api/videos/{id}/pose` | 逐帧关键点 + 置信度 + 时间戳 |
| GET | `/api/videos/{id}/swings/{k}/compare.jpg?cp=<检查点>&ref=<参考id>` | 关键帧对照图（左：你，右：最接近的职业挥拍，默认取最接近的那条） |
| GET | `/api/videos/{id}/file` | 转码后的视频 |
| GET | `/api/videos/{id}/corrections` | 纠错日志 |
| GET/PATCH/DELETE | `/api/references[/{id}]` | 参考条目：动作、审核状态（demo/coach_reviewed 需要审核人/rejected）、风格、启用 |
| GET | `/api/annot/videos`、`/api/annot/video?key=` | 可标注的视频（`data/raw/eval_real/`、`data/raw/annotate/`）及其标注 |
| PUT | `/api/annot/labels?key=` | 保存一个标注人的标注 |
| POST | `/api/annot/import/upload`、`/api/annot/import/youtube` | 导入本地视频 / 单个公开 YouTube 视频（可截取时间段），转换为标注格式，可选同时分析；返回任务，`GET /api/annot/jobs/{id}` 查进度 |
| POST | `/api/annot/from_video/{id}` | 为分析页里的视频生成标注副本 |
| GET | `/api/annot/suggestions?key=` | 系统检出的挥拍（换算成标注副本的帧号，仅作提示） |
| DELETE | `/api/annot/video?key=&labels=&app_video=` | 删除导入的视频（可选同时删除标注、对应分析）；评测集视频不能在这里删除 |

## 目录

```
src/tennis_coach/   video.py 视频校验/转码/时间戳/剪辑检测 · pose.py 姿态+跟踪+缓存 · skeleton.py 清洗/归一化/镜像
                    view.py 机位 · swing.py 挥拍/持拍手/阶段 · classify.py 识别 · features.py 指标 · dtw.py · match.py
                    feedback.py · quality.py · analysis.py 总流程 · db.py · service.py 任务队列 · api.py · web/ 前端
scripts/            数据获取、评测、调试
tests/              单元测试
docs/               DATASETS.md 数据核验清单 · MODELS.md 模型与算法 · REPORT.md 评测报告 · RESHOOT.md 补拍清单
models/             RTMPose-m / YOLOX-m ONNX 权重（Git LFS）
refpack/            职业参考包（Git LFS，只有派生数据，没有视频画面）
data/ (git 忽略)    raw/ 原始数据 · store/ 上传视频 · derived/ 姿态缓存与派生文件 · labels/ 评测标注 · *.sqlite3
```
