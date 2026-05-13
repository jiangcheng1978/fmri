#!/usr/bin/env python3
"""
fMRI 数据自动处理主流程
一键完成: DICOM转换 -> 预处理 -> 质量评估 -> 报告生成
用法:
    source .venv/bin/activate
    python scripts/run_pipeline.py
"""
import sys
from pathlib import Path
import subprocess

FMRI_ROOT = Path(__file__).parents[1]
PYTHON = sys.executable


def run_command(cmd, label=""):
    """运行命令并显示输出"""
    print(f"\n{'='*60}")
    print(f"[PIPELINE] {label}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {label} 执行失败")
        sys.exit(1)


def main():
    print(f"\n{'#'*60}")
    print(f"  fMRI 自动处理管线")
    print(f"  根目录: {FMRI_ROOT}")
    print(f"{'#'*60}")

    # 检查输入数据
    sub_003_dir = FMRI_ROOT / "data" / "sub_003"
    t1_dir = FMRI_ROOT / "data" / "t1_original"

    if not sub_003_dir.exists():
        print(f"[ERROR] 未找到 fMRI 数据目录: {sub_003_dir}")
        sys.exit(1)
    if not t1_dir.exists():
        print(f"[ERROR] 未找到 T1 数据目录: {t1_dir}")
        sys.exit(1)

    dicom_files = list(sub_003_dir.glob("*.IMA"))
    if not dicom_files:
        dicom_files = list(sub_003_dir.glob("*.dcm"))
    if not dicom_files:
        print("[ERROR] sub_003 中未找到 .IMA 或 .dcm 文件")
        sys.exit(1)

    print(f"\n[INFO] 找到 {len(dicom_files)} 个 fMRI DICOM 文件")
    print(f"[INFO] T1 DICOM 文件: {len(list(t1_dir.glob('*.IMA')))} 个")

    # 清理旧的 NIfTI 输出
    nifti_fmri = FMRI_ROOT / "output" / "nifti_fmri"
    nifti_t1 = FMRI_ROOT / "output" / "nifti_t1"
    nifti_fmri.mkdir(parents=True, exist_ok=True)
    nifti_t1.mkdir(parents=True, exist_ok=True)
    # 删除旧文件 (保留 JSON)
    for f in nifti_fmri.glob("*.nii.gz"):
        if not f.stem.endswith("json"):
            f.unlink()
    for f in nifti_t1.glob("*.nii.gz"):
        if not f.stem.endswith("json"):
            f.unlink()

    # Step 1: DICOM -> NIfTI 转换
    run_command([
        PYTHON, str(FMRI_ROOT / "scripts" / "dicom" / "dicom_convert.py"),
    ])

    # Step 2: fMRI 预处理
    run_command([
        PYTHON, str(FMRI_ROOT / "scripts" / "preprocess" / "fmri_preprocess.py"),
    ])

    # Step 3: 质量评估与报告生成
    run_command([
        PYTHON, str(FMRI_ROOT / "scripts" / "analysis" / "report_generator.py"),
    ])

    print(f"\n{'#'*60}")
    print(f"  全部完成!")
    print(f"{'#'*60}")
    print(f"\n输出文件:")
    print(f"  DICOM NIfTI:  {FMRI_ROOT / 'output' / 'nifti_fmri'}")
    print(f"  T1 NIfTI:     {FMRI_ROOT / 'output' / 'nifti_t1'}")
    print(f"  预处理数据:   {FMRI_ROOT / 'output' / 'output_fsl'}")
    print(f"  分析报告:     {FMRI_ROOT / 'output' / 'output_report' / 'fmri_analysis_report.md'}")
    print(f"  空间可视化:   {FMRI_ROOT / 'output' / 'output_report' / 'spatial_maps.png'}")
    print(f"  时间分析:     {FMRI_ROOT / 'output' / 'output_report' / 'temporal_analysis.png'}")


if __name__ == "__main__":
    main()
