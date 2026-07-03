# Ascend Operator Performance Collector

自动多模式采集 Ascend（昇腾）NPU 算子性能数据的 Python 脚本。

## 功能特色

- 支持 **pytest** 与 **Python 脚本** 两种测试文件自动识别与运行
- 三种编译模式自动切换测试：
  - **ssbuffer**：默认参数
  - **cvpipeline**：禁用 `enable_dynamic_cv_pipeline`
  - **native**：禁用动态流水线且 `set_workspace_multibuffer=0`
- 自动检测 `SSBUFFER` 回退，避免重复测试
- 当内嵌 profiling 失败时，自动使用 `msprof` 回退采集
- 设备自动重试：遇到 NPU 错误或 profiling 失败，自动尝试下一张卡
- 增量保存结果，每完成一个算子立即写入 CSV，防止数据丢失
- 输出宽表 `CSV`，包含各模式下的 `Avg Time(us)` 和详细备注

## 快速开始

```bash
# 全量测试所有 test_*_npu.py（自动适配 pytest / python 模式）
python auto_profile_collect_v3.11.py --pattern "**/test_*_npu.py" --perf-marker ""

# 仅测试特定算子（使用目录匹配）
python auto_profile_collect_v3.11.py --kernels chunk_fwd_kernel_h_split chunk_bwd_kernel_dh_split

# 跳过 pytest 文件，仅测试纯 Python 脚本
python auto_profile_collect_v3.11.py --pattern "**/test_*_npu.py" --skip-pytest

# 关闭 msprof 回退
python auto_profile_collect_v3.11.py --pattern "**/test_*_npu.py" --no-msprof-fallback
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--test-dir` | 测试文件根目录 | `.` |
| `--pattern` | 测试文件匹配模式（支持递归 `**`） | `test_*_perf.py` |
| `--kernels` | 指定算子名列表（启用目录匹配模式） | 无 |
| `--perf-marker` | pytest `-k` 筛选条件，设为空字符串取消筛选 | `perf` |
| `--output` | 输出 CSV 文件名 | `perf_summary_v3.csv` |
| `--modes` | 要测试的编译模式，可多选 | `ssbuffer cvpipeline native` |
| `--devices` | 手动指定 NPU 设备列表 | 自动从 `npu-smi info` 获取 |
| `--skip-pytest` | 跳过所有 pytest 文件 | 否 |
| `--no-msprof-fallback` | 禁用 msprof 自动回退 | 否（默认启用） |

## 工作流程

1. **文件发现**：根据 `--pattern` 或 `--kernels` 查找测试文件（递归）
2. **分类**：检测文件内容，分为 pytest 文件和 Python 脚本
3. **逐文件/批量测试**：
   - 对每个编译模式（ssbuffer/cvpipeline/native），动态修改 Triton 编译器参数
   - 运行测试（Python 脚本直接 `python`，pytest 批量 `pytest -k`）
   - 若 python 脚本无 profiling 数据，自动用 `msprof` 重新运行
4. **数据提取**：从 profiling 输出目录中读取 `op_statistic.csv`，提取 `Ratio(%)` 最高的算子的 `Avg Time(us)`
5. **结果保存**：每处理完一个文件，增量保存宽表 CSV 和原始记录 CSV
6. **异常处理**：自动换卡重试，备注记录失败原因

## 输出格式

生成的 CSV 文件包含如下列：
- `test_case`：测试用例名（文件名 stem）
- `ssbuffer_avg_time_us`, `cvpipeline_avg_time_us`, `native_avg_time_us`：三种模式下的平均耗时
- `fallback`：是否发生 ssbuffer 回退
- `ssbuffer_备注`, `cvpipeline_备注`, `native_备注`：各模式备注（如“msprof 采集”、“CSV 解析失败”等）

## 环境要求

- Python 3.11+
- PyTorch (Ascend 版本)
- Triton (Ascend 后端)
- pandas
- Ascend NPU 驱动及 `npu-smi` 工具
- `msprof` 命令行工具（可选，用于回退采集）




git add .
git commit -m "feat: 初始提交 - 多模式算子性能采集工具 v3.11

- 支持 pytest 和 Python 脚本自动识别
- 三种编译模式自动切换测试
- ssbuffer 回退检测
- msprof 自动回退采集
- 设备自动重试与增量保存
- 详细备注与宽表 CSV 输出"
git push -u origin main
```

如果遇到 `main` 分支不存在，先执行：
```bash
git branch -M main
git push -u origin main
```

---

### 7. 脚本流程概述（已写入 README，也可单独作为文档）

已在 README 的 “工作流程” 章节说明，此处不再重复。

---

完成以上步骤后，你的 GitHub 仓库即可公开分享，他人可以通过 `git clone` 获取并使用该脚本。