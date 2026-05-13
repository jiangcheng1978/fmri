#!/usr/bin/env python3
"""
DICOM -> NIfTI 转换脚本
支持西门子 MOSAIC 多band fMRI 数据自动解包
"""
import os
import sys
import shutil
import subprocess
import glob
from pathlib import Path


def get_dcm2niix_bin():
    """查找 dcm2niix 二进制文件"""
    # 1. 使用 Python 包中的路径
    try:
        import dcm2niix
        return str(dcm2niix.bin)
    except ImportError:
        pass

    # 2. 检查系统 PATH
    path = shutil.which("dcm2niix")
    if path:
        return path

    # 3. 检查常见安装位置
    for p in ["/opt/homebrew/bin/dcm2niix", "/usr/local/bin/dcm2niix"]:
        if os.path.exists(p):
            return p

    print("错误: 找不到 dcm2niix，请先安装 (pip install dcm2niix)")
    sys.exit(1)


def convert_dicom(input_dir, output_dir, subject_id="sub-003"):
    """
    将 DICOM 文件转换为 NIfTI 格式

    Args:
        input_dir: DICOM 文件目录
        output_dir: 输出目录
        subject_id: 被试标识
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dicom_bin = get_dcm2niix_bin()
    print(f"[dcm2niix] 使用二进制: {dicom_bin}")

    # 检查输入目录
    dicom_files = list(input_dir.glob("*.IMA"))
    if not dicom_files:
        dicom_files = list(input_dir.glob("*.dcm"))
    if not dicom_files:
        print(f"[ERROR] 目录 {input_dir} 中未找到 DICOM 文件")
        sys.exit(1)

    print(f"[dcm2niix] 找到 {len(dicom_files)} 个 DICOM 文件")
    print(f"[dcm2niix] 输出目录: {output_dir}")

    # 运行 dcm2niix 转换
    # -z: gzip 压缩输出
    # -o: 输出目录
    # -b: 不产生单独的二进制文件
    # -f: 输出文件名格式
    cmd = [
        dicom_bin,
        "-z", "y",          # gzip 压缩
        "-o", str(output_dir),
        "-b", "y",          # 单独保存 JSON
        "-f", f"{subject_id}_%s",
        str(input_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] dcm2niix 失败: {result.stderr}")
        sys.exit(1)

    # dcm2niix 输出的文件名格式是 subject + _ + number，需要重命名为更有意义的名称
    # 读取 JSON 中的系列信息来重命名
    import json as jsonmod
    nii_files = sorted(output_dir.glob("*.nii.gz"))
    for nii_f in nii_files:
        json_f = output_dir / (nii_f.stem + ".json")
        if json_f.exists():
            try:
                with open(json_f, "r") as fj:
                    meta = jsonmod.load(fj)
                series_desc = meta.get("SeriesDescription", "").replace(" ", "_").replace("-", "_")
                sequence_name = meta.get("SequenceName", "").replace(" ", "_").replace("*", "")
                # 确定类型
                modality = meta.get("Modality", "MR")
                if modality == "MR" and "bold" in series_desc.lower():
                    suffix = "bold"
                elif modality == "MR" and "mprage" in series_desc.lower():
                    suffix = "T1w"
                elif modality == "MR" and "ep2d_bold" in series_desc.lower():
                    suffix = "bold"
                else:
                    suffix = "anat" if series_desc == "" else series_desc

                if sequence_name:
                    new_name = f"{subject_id}_{sequence_name}_{suffix}.nii.gz"
                else:
                    new_name = f"{subject_id}_{series_desc}_{suffix}.nii.gz"

                new_json_name = f"{subject_id}_{series_desc}_{suffix}.json"

                nii_f.rename(output_dir / new_name)
                if json_f.exists():
                    json_f.rename(output_dir / new_json_name)

            except Exception as e:
                print(f"[WARN] 重命名失败: {e}")

    # 列出输出文件
    nii_files = list(output_dir.glob("*.nii.gz"))
    json_files = list(output_dir.glob("*.json"))
    print(f"[dcm2niix] 转换完成!")
    print(f"[dcm2niix] 生成 {len(nii_files)} 个 .nii.gz 文件")
    print(f"[dcm2niix] 生成 {len(json_files)} 个 .json 元数据文件")

    for f in sorted(nii_files):
        print(f"  - {f.name}")

    return nii_files, json_files


def validate_conversion(nii_files, expected_modality):
    """验证转换结果"""
    import nibabel as nib

    print("\n[dcm2niix] 验证转换结果...")
    for nii_file in nii_files:
        img = nib.load(nii_file)
        data = img.get_fdata()
        if expected_modality == "fmri":
            print(f"  {nii_file.name}: shape={data.shape}, dtype={data.dtype}")
        else:
            print(f"  {nii_file.name}: shape={data.shape}, dtype={data.dtype}")
    print("[dcm2niix] 验证完成")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DICOM -> NIfTI 转换")
    parser.add_argument("--input", type=str, default=None,
                        help="DICOM 输入目录（默认自动检测）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录（默认自动）")
    parser.add_argument("--subject", type=str, default="sub-003",
                        help="被试 ID")
    args = parser.parse_args()

    # 自动检测路径
    fmri_root = Path(__file__).parents[2]
    data_dir = fmri_root / "data"

    if args.input:
        fmri_input = Path(args.input)
        t1_input = Path(args.input)  # 需要手动指定
    else:
        fmri_input = data_dir / "sub_003"
        t1_input = data_dir / "t1_original"

    if args.output:
        fmri_output = Path(args.output)
        t1_output = Path(args.output)
    else:
        fmri_output = fmri_root / "output" / "nifti_fmri"
        t1_output = fmri_root / "output" / "nifti_t1"

    # 转换 T1 结构像
    print("=" * 60)
    print("1. 转换 T1 结构像")
    print("=" * 60)
    if t1_input.exists():
        convert_dicom(t1_input, t1_output, subject_id="sub-003_t1")
        print()

    # 转换 fMRI 数据
    print("=" * 60)
    print("2. 转换 fMRI BOLD 数据")
    print("=" * 60)
    if fmri_input.exists():
        nii_files, json_files = convert_dicom(fmri_input, fmri_output, subject_id="sub-003")
        validate_conversion(nii_files, "fmri")
        print()

    print("[INFO] DICOM 转换全部完成!")
    print(f"[INFO] fMRI NIfTI 文件: {fmri_output}")
    print(f"[INFO] T1 NIfTI 文件: {t1_output}")
