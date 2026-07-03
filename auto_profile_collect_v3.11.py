#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本3 V3.11：增强调试与 msprof 路径修复
=========================================
- 添加 --skip-pytest 选项
- msprof 输出目录内容打印，便于定位 CSV
- 统一保存点，避免文件覆盖
- dropna=False 保留所有 test_case
- 执行命令：放在fla的cv_operator下，python auto_profile_collect_v3.11.py --pattern "**/test_*_npu.py" --perf-marker ""
"""

import os, re, sys, shutil, shlex, subprocess, argparse, glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd

# -------------------- 配置 --------------------
COMPILER_PY = Path("/usr/local/python3.11.13/lib/python3.11/site-packages/triton/backends/ascend/compiler.py")
BACKUP_EXT = ".bak_v3"

BASE_ENV = {
    "TRITON_DEBUG": "1",
    "TRITON_DISABLE_AUTOTUNE": "1",
    "TRITON_ALWAYS_COMPILE": "1",
}

MODE_PARAMS = {
    "ssbuffer":   (None,  None),
    "cvpipeline": (False, None),
    "native":     (False, 0),
}

# -------------------- 工具函数 --------------------
def get_available_devices():
    try:
        res = subprocess.run("npu-smi info -l", shell=True, capture_output=True, text=True, timeout=5)
        ids = re.findall(r"NPU\s+(\d+)\s+", res.stdout)
        if ids: return [int(x) for x in ids]
    except: pass
    return [0, 1, 2, 3]

def is_pytest_file(file_path: Path):
    try:
        content = file_path.read_text()
        return bool(re.search(r'\bimport pytest\b|\bfrom pytest\b|@pytest\.mark\.', content))
    except: return False

def run_single_file_python(file_path: Path, device: int, use_msprof: bool = False, prof_output_dir: str = None):
    env = {**os.environ, **BASE_ENV, "ASCEND_RT_VISIBLE_DEVICES": str(device)}
    work_dir = file_path.parent
    cache = Path.home() / ".triton" / "cache"
    if cache.exists(): shutil.rmtree(cache, ignore_errors=True)

    if use_msprof:
        # 使用绝对路径避免歧义
        abs_prof_dir = Path(prof_output_dir).resolve()
        cmd = f"msprof --output={shlex.quote(str(abs_prof_dir))} python {shlex.quote(file_path.name)}"
    else:
        cmd = f"python {shlex.quote(str(file_path.name))}"

    try:
        return subprocess.run(cmd, shell=True, cwd=work_dir,
                             capture_output=True, text=True, timeout=1800, env=env)
    except subprocess.TimeoutExpired:
        return None

def run_pytest_batch(test_files, perf_marker, device):
    env = {**os.environ, **BASE_ENV, "ASCEND_RT_VISIBLE_DEVICES": str(device)}
    work_dir = test_files[0].parent
    file_list = " ".join(shlex.quote(str(f.name)) for f in test_files)
    cmd = f"pytest -sv -k '{perf_marker}' {file_list}" if perf_marker else f"pytest -sv {file_list}"
    cache = Path.home() / ".triton" / "cache"
    if cache.exists(): shutil.rmtree(cache, ignore_errors=True)
    try:
        return subprocess.run(cmd, shell=True, cwd=work_dir,
                             capture_output=True, text=True, timeout=1800, env=env)
    except subprocess.TimeoutExpired:
        return None

# -------------------- 编译器修改 --------------------
def backup_compiler(): shutil.copy2(COMPILER_PY, COMPILER_PY.with_suffix(BACKUP_EXT))
def restore_compiler():
    bak = COMPILER_PY.with_suffix(BACKUP_EXT)
    if bak.exists(): shutil.copy2(bak, COMPILER_PY)

def modify_compiler(enable_dynamic, workspace_multibuffer):
    with open(COMPILER_PY, "r") as f: lines = f.readlines()
    new_lines = []
    for line in lines:
        if enable_dynamic is not None and "enable_dynamic_cv_pipeline" in line:
            line = re.sub(r'(enable_dynamic_cv_pipeline:\s*bool\s*=\s*).*', r'\g<1>'+str(enable_dynamic), line)
        if workspace_multibuffer is not None and "set_workspace_multibuffer" in line:
            line = re.sub(r'(set_workspace_multibuffer:\s*int\s*=\s*).*', r'\g<1>'+str(workspace_multibuffer), line)
        new_lines.append(line)
    with open(COMPILER_PY, "w") as f: f.writelines(new_lines)

# -------------------- 文件发现 --------------------
def find_test_files_by_pattern(test_dir, pattern): return sorted(Path(test_dir).rglob(pattern))
def find_test_files_by_kernels(kernels, test_dir, kernel_pattern):
    base = Path(test_dir)
    files = []
    for k in kernels:
        d = base / k if (base/k).is_dir() else next((d for d in base.iterdir() if d.is_dir() and d.name.startswith(k)), None)
        if d: files.extend(sorted(d.glob(kernel_pattern)))
    return files

# -------------------- 输出解析 --------------------
def extract_profiling_dirs(output): return [Path(m) for m in re.findall(r"Start parsing profiling data in sync mode at:\s*(\S+)", output) if Path(m).exists()]
def extract_test_case_names(output): return re.findall(r"PASSED\s+(\S+)\s", output)

def extract_avg_time_from_csv(csv_path: Path) -> Optional[float]:
    if not csv_path.exists():
        print(f"[调试] CSV 文件不存在: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty or "Avg Time(us)" not in df.columns:
            print(f"[调试] CSV 为空或缺少列: {csv_path}")
            return None
        if "Ratio(%)" in df.columns:
            df = df.dropna(subset=["Ratio(%)"])
            if not df.empty:
                best = df.loc[df["Ratio(%)"].idxmax()]
                return best["Avg Time(us)"]
        best = df.loc[df["Avg Time(us)"].idxmax()]
        return best["Avg Time(us)"]
    except Exception as e:
        print(f"[调试] 读取 CSV 异常: {e}")
        return None

def find_latest_msprof_csv(prof_output_dir: str) -> Optional[Path]:
    """查找 msprof 输出的最新 op_statistic CSV，并打印调试信息"""
    base = Path(prof_output_dir)
    print(f"[msprof 调试] 查找目录: {base} (存在: {base.exists()})")
    if not base.exists():
        return None
    prof_dirs = sorted(base.glob("PROF_*"), key=os.path.getmtime, reverse=True)
    print(f"[msprof 调试] 找到 {len(prof_dirs)} 个 PROF_* 目录")
    for pd in prof_dirs:
        csv_files = list(pd.glob("mindstudio_profiler_output/op_statistic_*.csv"))
        print(f"[msprof 调试]   {pd.name}: mindstudio_profiler_output 下 CSV 文件: {[f.name for f in csv_files]}")
        if csv_files:
            chosen = sorted(csv_files, key=os.path.getmtime, reverse=True)[0]
            print(f"[msprof 调试] 选中 CSV: {chosen}")
            return chosen
    return None

def check_ssbuffer_fallback(output): return "SSBUFFER return code=2, will fallback to enable_dynamic_cv_pipeline=False" in output

def save_progress(raw_records, output_path):
    """增量保存，强制保留所有 test_case，并输出最终文件内容校验"""
    if not raw_records:
        print("[调试] 无记录，跳过保存")
        return
    df = pd.DataFrame(raw_records)
    df['test_case'] = df['test_case'].astype(str).str.strip()
    print(f"[保存调试] 当前 raw_records 行数: {len(df)}, 唯一 test_case: {df['test_case'].nunique()}")
    print(f"            所有 test_case: {df['test_case'].unique().tolist()}")

    raw_path = output_path.replace('.csv', '_raw.csv')
    df.to_csv(raw_path, index=False, encoding='utf-8-sig')

    try:
        pivot = df.pivot_table(index="test_case", columns="mode", values="avg_time_us",
                               aggfunc="first", dropna=False)
        print(f"[保存调试] pivot 行数: {len(pivot)}, test_case: {pivot.index.tolist()}")
        if not pivot.empty:
            pivot.columns = [f"{mode}_avg_time_us" for mode in pivot.columns]
            pivot.reset_index(inplace=True)
            fallback_series = df[df["mode"] == "ssbuffer"].set_index("test_case")["fallback"]
            pivot["fallback"] = pivot["test_case"].map(fallback_series)
            notes = df.groupby(["test_case", "mode"])["备注"].first().unstack(fill_value="")
            notes.columns = [f"{mode}_备注" for mode in notes.columns]
            pivot = pivot.merge(notes.reset_index(), on="test_case", how="left")

            # 强制写入并立即关闭
            pivot.to_csv(output_path, index=False, encoding='utf-8-sig')
            # 验证写入后的文件内容
            check = pd.read_csv(output_path)
            print(f"[保存调试] 写入后 CSV 行数: {len(check)}, test_case: {check['test_case'].tolist()}")
        else:
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print("[保存调试] 宽表为空，直接保存原始记录")
    except Exception as e:
        print(f"[错误] 生成宽表失败: {e}")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

# -------------------- 主流程 --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default=".")
    parser.add_argument("--pattern", default="test_*_perf.py")
    parser.add_argument("--kernels", nargs="*")
    parser.add_argument("--kernel-pattern", default="test_*_npu.py")
    parser.add_argument("--perf-marker", default="perf")
    parser.add_argument("--output", default="perf_summary_v3.csv")
    parser.add_argument("--modes", nargs="+", default=list(MODE_PARAMS.keys()))
    parser.add_argument("--devices", nargs="+", type=int)
    parser.add_argument("--keep-backup", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true", help="跳过 pytest 文件，仅处理 python 文件")
    parser.add_argument("--no-msprof-fallback", dest="msprof_fallback", action="store_false",
                        help="禁用 msprof 自动回退")
    args = parser.parse_args()

    devices = args.devices or get_available_devices()
    print(f"可用设备: {devices}")

    if args.kernels:
        test_files = find_test_files_by_kernels(args.kernels, args.test_dir, args.kernel_pattern)
    else:
        test_files = find_test_files_by_pattern(args.test_dir, args.pattern)

    if not test_files:
        pd.DataFrame([{"test_case": "NO_FILES"}]).to_csv(args.output, index=False)
        return

    pytest_files = [f for f in test_files if is_pytest_file(f)]
    python_files = [f for f in test_files if not is_pytest_file(f)]
    print(f"Pytest 文件 ({len(pytest_files)}), Python 文件 ({len(python_files)})")
    if args.skip_pytest:
        pytest_files = []
        print("已跳过 pytest 文件")

    if not COMPILER_PY.exists():
        sys.exit("compiler.py 未找到")
    backup_compiler()

    raw_records = []
    msprof_output_base = Path(args.test_dir).resolve() / "msprof_tmp"
    msprof_output_base.mkdir(parents=True, exist_ok=True)

    try:
        # ---------- pytest 批量处理 ----------
        if pytest_files:
            print("开始处理 pytest 文件...")
            for mode in args.modes:
                restore_compiler()
                if mode == "cvpipeline": modify_compiler(False, None)
                elif mode == "native": modify_compiler(False, 0)
                result = run_pytest_batch(pytest_files, args.perf_marker, devices[0])
                if result:
                    output = result.stdout + "\n" + result.stderr
                    prof_dirs = extract_profiling_dirs(output)
                    case_names = extract_test_case_names(output)
                    print(f"[pytest] mode={mode}, PASSED 用例数: {len(case_names)}, profiling 目录数: {len(prof_dirs)}")
                    for i, case in enumerate(case_names):
                        prof_dir = prof_dirs[i] if i < len(prof_dirs) else None
                        avg = extract_avg_time_from_csv(prof_dir / "ASCEND_PROFILER_OUTPUT/op_statistic.csv") if prof_dir else None
                        raw_records.append({
                            "test_case": case, "mode": mode,
                            "avg_time_us": avg,
                            "备注": "" if avg else ("无 profiling 目录" if not prof_dir else "CSV 解析失败"),
                            "fallback": False
                        })
            # 不在这里保存，等 python 文件处理完再统一保存，避免覆盖

        # ---------- python 文件处理 ----------
        for file in python_files:
            print(f"\n========== {file.name} ==========")
            for mode in args.modes:
                restore_compiler()
                if mode == "cvpipeline": modify_compiler(False, None)
                elif mode == "native": modify_compiler(False, 0)

                success = False
                avg = None
                note = ""
                # 常规运行
                for dev in devices:
                    res = run_single_file_python(file, dev)
                    if res is None: continue
                    output = res.stdout + "\n" + res.stderr
                    if "NPU error" in output or "RuntimeError" in output:
                        continue
                    prof_dirs = extract_profiling_dirs(output)
                    if prof_dirs:
                        avg = extract_avg_time_from_csv(prof_dirs[-1] / "ASCEND_PROFILER_OUTPUT/op_statistic.csv")
                        note = "" if avg else "CSV 解析失败"
                    else:
                        note = "无 profiling 目录"
                    fallback = check_ssbuffer_fallback(output) if mode == "ssbuffer" else False
                    raw_records.append({
                        "test_case": file.stem, "mode": mode,
                        "avg_time_us": avg,
                        "备注": note,
                        "fallback": fallback
                    })
                    success = True
                    break

                # msprof 回退
                if (not success) or (success and avg is None):
                    if args.msprof_fallback:
                        print(f"  [{mode}] 尝试 msprof 采集...")
                        prof_out = str(msprof_output_base / f"{file.stem}_{mode}")
                        res = run_single_file_python(file, devices[0], use_msprof=True, prof_output_dir=prof_out)
                        if res:
                            output = res.stdout + "\n" + res.stderr
                            fallback = check_ssbuffer_fallback(output) if mode == "ssbuffer" else False
                            csv_path = find_latest_msprof_csv(prof_out)
                            avg_ms = extract_avg_time_from_csv(csv_path) if csv_path else None
                            ms_note = "msprof 采集" if avg_ms else ("msprof 采集失败" if csv_path else "无 msprof 数据")
                            if success:
                                # 更新已有记录
                                raw_records[-1]["avg_time_us"] = avg_ms
                                raw_records[-1]["备注"] = ms_note
                                raw_records[-1]["fallback"] = fallback
                            else:
                                raw_records.append({
                                    "test_case": file.stem, "mode": mode,
                                    "avg_time_us": avg_ms,
                                    "备注": ms_note,
                                    "fallback": fallback
                                })
                        else:
                            if not success:
                                raw_records.append({
                                    "test_case": file.stem, "mode": mode,
                                    "avg_time_us": None,
                                    "备注": "msprof 运行失败",
                                    "fallback": False
                                })
                elif not success:
                    raw_records.append({
                        "test_case": file.stem, "mode": mode,
                        "avg_time_us": None,
                        "备注": "所有设备均失败",
                        "fallback": False
                    })

            # 每个文件处理完后保存一次
            save_progress(raw_records, args.output)

    finally:
        restore_compiler()
        bak = COMPILER_PY.with_suffix(BACKUP_EXT)
        if not args.keep_backup and bak.exists(): os.remove(bak)

    save_progress(raw_records, args.output)
    print(f"\n全部完成！最终结果: {args.output}")

if __name__ == "__main__":
    main()