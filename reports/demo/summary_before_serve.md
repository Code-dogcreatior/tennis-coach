# 闭环演示结果（自动生成）

参考视频 36 个 → 参考挥拍 197 条，处理 118.3 s（含转码+姿态）；查询 19 个，处理 70.9 s。

| 查询文件 | 真值 | 识别（主挥拍） | 置信度 | 持拍手（来源/置信度） | 机位 | 质量 | 匹配 | 反馈条数 |
|---|---|---|---|---|---|---|---|---|
| Rw3ZDfTTD9M.mp4 | forehand (left-handed, multiple) | forehand | 0.98 | left（estimated/1.00；真值 left） | 正面 | good | ok | 0 |
| p7_fvolley_s1.avi | forehand_volley | forehand | 0.91 | right（estimated/1.00；真值 right） | 正面 | warn | ok | 2 |
| p7_smash_s1.avi | smash | unknown | 0.40 | right（estimated/1.00；真值 right） | 正面 | warn | action_unknown | 0 |
| p7_serflat_s1.avi | flat_service | serve | 0.60 | right（estimated/0.80；真值 right） | 斜前方 | warn | ok | 0 |
| p7_backhand2h_s1.avi | backhand2hands | backhand_2h | 0.98 | right（default_right_unverified/0.00；真值 right） | 斜前方 | warn | ok | 0 |
| p7_backhand_s1.avi | backhand | backhand_1h | 0.91 | right（estimated/1.00；真值 right） | 正面 | warn | ok | 0 |
| p7_foreflat_s1.avi | forehand_flat | forehand | 0.98 | right（estimated/1.00；真值 right） | 正面 | warn | ok | 0 |
| p35_fvolley_s1.avi | forehand_volley | backhand_1h | 0.71 | left（estimated/1.00；真值 left） | 正面 | warn | ok | 2 |
| p35_smash_s1.avi | smash | serve | 0.60 | left（estimated/0.80；真值 left） | 斜前方 | warn | ok | 0 |
| p35_serflat_s1.avi | flat_service | serve | 0.42 | left（estimated/0.80；真值 left） | 侧面（非持拍手一侧） | warn | ok | 2 |
| p35_backhand2h_s1.avi | backhand2hands | unknown | 0.40 | right（default_right_unverified/0.00；真值 left） | 正面 | warn | action_unknown | 0 |
| p35_backhand_s1.avi | backhand | backhand_2h | 0.98 | left（estimated/0.57；真值 left） | 正面 | warn | ok | 2 |
| p35_foreflat_s1.avi | forehand_flat | forehand | 0.91 | left（estimated/1.00；真值 left） | 正面 | warn | ok | 0 |
| p3_fvolley_s1.avi | forehand_volley | unknown | 0.40 | right（default_right_unverified/0.00；真值 left） | 斜前方 | warn | action_unknown | 0 |
| p3_smash_s1.avi | smash | unknown | 0.40 | left（estimated/1.00；真值 left） | 斜前方 | warn | action_unknown | 0 |
| p3_serflat_s1.avi | flat_service | serve | 0.60 | left（estimated/0.80；真值 left） | 斜前方 | warn | ok | 0 |
| p3_backhand2h_s1.avi | backhand2hands | backhand_1h | 0.91 | left（estimated/1.00；真值 left） | 正面 | warn | ok | 1 |
| p3_backhand_s1.avi | backhand | backhand_1h | 0.98 | left（estimated/1.00；真值 left） | 正面 | warn | ok | 2 |
| p3_foreflat_s1.avi | forehand_flat | unknown | 0.40 | right（default_right_unverified/0.00；真值 left） | 正面 | warn | action_unknown | 0 |

## 与职业动作对比（主挥拍）

| 查询文件 | 识别 | 最接近的职业挥拍 | 主要问题（已验证检查点） | 参考性提示 |
|---|---|---|---|---|
| Rw3ZDfTTD9M.mp4 | forehand | Novak Djokovic | 无 | – |
| p7_fvolley_s1.avi | forehand | Novak Djokovic | 随挥不完整，收拍过低（你 0.22，职业 0.98–1.67）；挥拍路径偏平或由上往下（你 0.40，职业 0.90–1.70） | – |
| p7_smash_s1.avi | unknown | – | – | |
| p7_serflat_s1.avi | serve | Ben Shelton | 发球击球点偏低（你 0.35，职业 0.85–1.09）；抛球手没有充分上举（你 0.68，职业 0.93–1.06） | 奖杯姿势时肩膀没有倾斜 |
| p7_backhand2h_s1.avi | backhand_2h | Alex Eala | 无 | – |
| p7_backhand_s1.avi | backhand_1h | Dominic Thiem | 无 | – |
| p7_foreflat_s1.avi | forehand | Novak Djokovic | 无 | – |
| p35_fvolley_s1.avi | backhand_1h | Dominic Thiem | 挥拍路径偏平或由上往下（你 0.00，职业 0.62–1.75）；随挥不完整，收拍过低（你 0.48，职业 0.79–1.63） | – |
| p35_smash_s1.avi | serve | Ben Shelton | 击球时抛球手没有回收（你 1.16，职业 -0.69–-0.01）；发球时向上蹬伸不足（你 -0.04，职业 0.22–1.04） | 奖杯姿势时肩膀没有倾斜；持拍手下落不充分 |
| p35_serflat_s1.avi | serve | – | 无 | – |
| p35_backhand2h_s1.avi | unknown | – | – | |
| p35_backhand_s1.avi | backhand_2h | Holger Rune | 击球时身体没有转过来，像是手臂单独挥拍（你 -0.17，职业 0.28–0.67） | – |
| p35_foreflat_s1.avi | forehand | Alex Eala | 无 | 随挥没有越过身体 |
| p3_fvolley_s1.avi | unknown | – | – | |
| p3_smash_s1.avi | unknown | – | – | |
| p3_serflat_s1.avi | serve | serve compilation (Fritz/Rublev/Alcaraz/Djokovic/Sinner/Zverev/Shelton...) | 发球击球点偏低（你 0.60，职业 0.85–1.09）；发球时向上蹬伸不足（你 -0.04，职业 0.22–1.04） | – |
| p3_backhand2h_s1.avi | backhand_1h | Dominic Thiem | 无 | – |
| p3_backhand_s1.avi | backhand_1h | Dominic Thiem | 无 | – |
| p3_foreflat_s1.avi | unknown | – | – | |
