#!/usr/bin/env python3
"""
fMRI 数据分析与质量评估
生成详细的质量评估报告和统计分析
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import nibabel as nib

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_data(filepath):
    """加载 NIfTI"""
    img = nib.load(filepath)
    return img, img.get_fdata()


def quality_metrics(data, motions=None):
    """
    计算关键质量指标
    注意: data 应该是预处理后的数据 (去趋势+滤波之后)
    tSNR = mean / std along time axis — 对预处理数据仍然有意义
    """
    metrics = {}

    # 1. 信号强度 (使用脑内体素)
    brain_vals = data[data > np.percentile(data, 5)] if data.size > 0 else data
    if len(brain_vals) == 0:
        brain_vals = data

    global_mean = float(np.mean(brain_vals))
    global_std = float(np.std(brain_vals))
    metrics["signal_mean"] = global_mean
    metrics["signal_std"] = global_std
    metrics["snr"] = global_mean / global_std if global_std > 0 else 0.0
    metrics["max_signal"] = float(np.max(data))
    metrics["min_signal"] = float(np.min(data))

    # 2. 时间信噪比 tSNR = 全脑均值 / 时间方向标准差 (每个体素)
    n_timepoints = data.shape[3] if data.ndim == 4 else 1
    if n_timepoints > 1:
        mean_3d = np.mean(data, axis=3)
        std_3d = np.std(data, axis=3)
        tsnr_3d = mean_3d / (std_3d + 1e-10)
        # 只考虑脑内体素 (tsNR > 0)
        tsnr_mask = tsnr_3d > 0
        if tsnr_mask.sum() > 0:
            metrics["tsnr_mean"] = float(np.mean(tsnr_3d[tsnr_mask]))
            metrics["tsnr_median"] = float(np.median(tsnr_3d[tsnr_mask]))
        else:
            # 如果全为负数(去趋势)，用绝对值
            metrics["tsnr_mean"] = float(np.mean(np.abs(tsnr_3d)))
            metrics["tsnr_median"] = float(np.median(np.abs(tsnr_3d)))
    else:
        metrics["tsnr_mean"] = 0.0
        metrics["tsnr_median"] = 0.0

    # 3. 信号漂移 (前后 1/4 时间窗的 std 差异)
    if n_timepoints > 4:
        signals = data.reshape(-1, n_timepoints)
        first_quarter = np.std(signals[:, :n_timepoints // 4], axis=1)
        last_quarter = np.std(signals[:, -n_timepoints // 4:], axis=1)
        drift_pct = abs(np.mean(last_quarter) - np.mean(first_quarter)) / (np.mean(first_quarter) + 1e-10) * 100
        metrics["signal_drift_pct"] = float(drift_pct)
    else:
        metrics["signal_drift_pct"] = 0.0

    # 4. 头动指标
    if motions is not None:
        motions = np.array(motions)
        motions = motions[~np.all(np.isnan(motions), axis=1)]
        if len(motions) > 0:
            metrics["max_trans_x"] = float(np.max(np.abs(motions[:, 0])))
            metrics["max_trans_y"] = float(np.max(np.abs(motions[:, 1])))
            metrics["max_trans_z"] = float(np.max(np.abs(motions[:, 2])))
            metrics["max_rot_rx"] = float(np.max(np.abs(motions[:, 3])))
            metrics["max_rot_ry"] = float(np.max(np.abs(motions[:, 4])))
            metrics["max_rot_rz"] = float(np.max(np.abs(motions[:, 5])))

            # FD
            fd = np.zeros(len(motions))
            for i in range(1, len(motions)):
                dd = np.diff(motions[:i+1, :3], axis=0)
                fd[i] = 0.5 * np.sum(np.abs(dd))
            positive_fd = fd[fd > 0]
            metrics["max_fd"] = float(np.max(positive_fd)) if len(positive_fd) > 0 else 0.0
            metrics["mean_fd"] = float(np.mean(fd))
            metrics["median_fd"] = float(np.median(fd))
            metrics["fd_above_0.5"] = int(np.sum(fd > 0.5))
            metrics["fd_above_0.2"] = int(np.sum(fd > 0.2))

    return metrics


def spatial_maps(data, output_dir):
    """生成空间分布图"""
    if not HAS_MPL:
        print("  [SKIP] 需要 matplotlib 生成功能图")
        return

    n_volumes = data.shape[3]

    # 平均激活图
    mean_img = np.mean(data, axis=3)
    z_idx = data.shape[2] // 2
    x_idx = data.shape[0] // 2
    y_idx = data.shape[1] // 2

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 1. 平均激活 - axial slice
    ax = axes[0]
    ax.imshow(mean_img[:, :, z_idx], cmap="gray", origin="lower",
              interpolation="nearest")
    ax.set_title(f"Mean Activation (z={z_idx})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # 2. 平均激活 - sagittal
    ax = axes[1]
    ax.imshow(mean_img[x_idx, :, :], cmap="gray", origin="lower",
              interpolation="nearest")
    ax.set_title(f"Mean Activation (x={x_idx})")
    ax.set_xlabel("Y")
    ax.set_ylabel("Z")

    # 3. 平均激活 - coronal
    ax = axes[2]
    ax.imshow(mean_img[:, y_idx, :], cmap="gray", origin="lower",
              interpolation="nearest")
    ax.set_title(f"Mean Activation (y={y_idx})")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")

    # 4. 信号强度图
    ax = axes[3]
    ax.plot(mean_img.flatten(), ".", markersize=1, alpha=0.5)
    ax.set_title(f"Signal Distribution\nμ={np.mean(mean_img):.1f}, σ={np.std(mean_img):.1f}")
    ax.set_xlabel("Voxel index")
    ax.set_ylabel("Signal intensity")

    plt.tight_layout()
    plt.savefig(output_dir / "spatial_maps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  已保存: spatial_maps.png")


def temporal_analysis(data_4d, tr=2.0, output_dir=None):
    """时间序列分析"""
    if not HAS_MPL:
        return

    n_timepoints = data_4d.shape[3]
    times = np.arange(n_timepoints) * tr

    # 计算全脑平均信号
    global_signal = np.nanmean(data_4d, axis=(0, 1, 2))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 1. 全脑平均信号
    ax = axes[0, 0]
    ax.plot(times, global_signal, linewidth=0.5)
    ax.set_title(f"Global Mean Signal (n={n_timepoints})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal")

    # 2. 功率谱密度
    ax = axes[0, 1]
    fft_vals = np.abs(np.fft.rfft(global_signal - np.mean(global_signal))) ** 2
    freqs = np.fft.rfftfreq(n_timepoints, d=tr)
    ax.plot(freqs, fft_vals)
    ax.set_title(f"Power Spectral Density (f_max={freqs[-1]:.2f} Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_xlim(0, 0.3)

    # 3. tSNR map (简化) — 用绝对值计算 tSNR (处理去趋势后数据)
    ax = axes[1, 0]
    z_idx = data_4d.shape[2] // 2
    z_slice = data_4d[:, :, z_idx, :]
    mean_z = np.mean(np.abs(z_slice), axis=-1)
    std_z = np.std(z_slice, axis=-1)
    tsnr_z = mean_z / (std_z + 1e-10)
    ax.imshow(tsnr_z, cmap="hot", origin="lower")
    ax.set_title(f"tSNR Map (z={z_idx})\nMean={np.mean(tsnr_z[tsnr_z>0]):.1f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # 4. 信号幅度直方图 (使用绝对值，因为去趋势后均值接近 0)
    ax = axes[1, 1]
    ax.hist(np.abs(data_4d).flatten(), bins=100, alpha=0.7, density=True)
    ax.set_title("Signal Magnitude Histogram")
    ax.set_xlabel("|Signal|")
    ax.set_ylabel("Density")

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "temporal_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  已保存: temporal_analysis.png")


def generate_report(fmri_data_dir, preprocess_dir, output_dir):
    """
    生成完整分析报告
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("fMRI 分析报告生成")
    print(f"{'='*60}")

    # 收集数据
    # 1. 检查预处理结果
    preproc_file = Path(preprocess_dir) / "sub-003_preproc.nii.gz"
    motion_file = Path(preprocess_dir) / "motion_params.txt"
    fd_file = Path(preprocess_dir) / "framewise_displacement.txt"
    stats_file = Path(preprocess_dir) / "preprocessing_stats.json"

    # 2. 加载运动参数
    motions = None
    if motion_file.exists():
        motions = np.loadtxt(motion_file)

    # 3. 加载预处理数据
    data_4d = None
    if preproc_file.exists():
        print("\n[分析] 加载预处理数据...")
        img, data = load_data(preproc_file)
        data_4d = data
        print(f"  加载成功: {preproc_file.name}")
    else:
        print("\n[警告] 预处理数据不存在，跳过质量评估")
        print("  请先运行: python scripts/preprocess/fmri_preprocess.py")

    # 4. 生成报告
    report_lines = []
    report_lines.append("# fMRI 数据分析报告")
    report_lines.append("")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # 数据基本信息
    report_lines.append("## 1. 数据基本信息")
    report_lines.append("")

    # 从元数据获取
    json_files = list(Path(fmri_data_dir).glob("*.json"))
    if json_files:
        with open(json_files[0], "r") as f:
            metadata = json.load(f)
        report_lines.append(f"- 序列描述: {metadata.get('SequenceDescription', metadata.get('SeriesDescription', 'N/A'))}")
        te = metadata.get('EchoTime', None)
        if te is not None:
            te_str = f"{te*1000:.0f}" if te < 0.1 else f"{te:.2f}"
            report_lines.append(f"- Echo Time (TE): {te_str} ms")
        tr = metadata.get('RepetitionTime', None)
        if tr is not None:
            tr_str = f"{tr*1000:.0f}" if tr < 1 else f"{tr:.3f}"
            report_lines.append(f"- Repetition Time (TR): {tr_str} ms ({tr:.3f} s)")
        else:
            report_lines.append("- Repetition Time (TR): N/A")
        report_lines.append(f"- Flip Angle: {metadata.get('FlipAngle', 'N/A')}°")
        report_lines.append(f"- Slice Thickness: {metadata.get('SliceThickness', 'N/A')} mm")
        rows = metadata.get('Rows', metadata.get('Columns', 'N/A'))
        cols = metadata.get('Columns', metadata.get('Rows', 'N/A'))
        report_lines.append(f"- 矩阵: {rows}x{cols}")
        report_lines.append(f"- 时间点数 (Volumes): {metadata.get('NumberOfVolumes', 'N/A')}")
        report_lines.append(f"- 场强: {metadata.get('MagneticFieldStrength', 'N/A')} T")
        report_lines.append(f"- 扫描仪型号: {metadata.get('ManufacturersModelName', 'N/A')}")

    report_lines.append("")

    # 预处理结果
    report_lines.append("## 2. 预处理结果")
    report_lines.append("")

    if stats_file.exists():
        with open(stats_file, "r") as f:
            stats = json.load(f)
        report_lines.append(f"- 预处理后数据: `sub-003_preproc.nii.gz`")
        report_lines.append(f"- 数据维度: {stats.get('shape', 'N/A')}")
        report_lines.append(f"- 时间点数: {stats.get('n_volumes', 'N/A')}")
        report_lines.append(f"- 层数: {stats.get('n_slices', 'N/A')}")
        report_lines.append(f"- TR: {stats.get('tr', 'N/A')}s")
        report_lines.append(f"- 总时长: {stats.get('total_duration', 'N/A')}s")
        report_lines.append(f"- 空间平滑: FWHM={stats.get('fwhm_smooth_mm', stats.get('fwhm_smooth', 'N/A'))}mm")
        bp = stats.get('bandpass_hz', stats.get('bandpass', 'N/A'))
        if isinstance(bp, list):
            report_lines.append(f"- 带通滤波: {bp[0]}-{bp[1]} Hz")
        else:
            report_lines.append(f"- 带通滤波: {bp} Hz")

        if "snr" in stats:
            report_lines.append(f"- 信噪比 (SNR): {stats.get('snr', 'N/A')}")
        if "tsnr_mean" in stats:
            report_lines.append(f"- 时间信噪比 (tSNR): {stats.get('tsnr_mean', 'N/A')}")
    else:
        report_lines.append("- 预处理数据尚未生成")
        report_lines.append("  运行命令: `python scripts/preprocess/fmri_preprocess.py`")

    report_lines.append("")

    # 质量评估
    report_lines.append("## 3. 质量评估")
    report_lines.append("")

    if data_4d is not None:
        metrics = quality_metrics(data_4d, motions)

        # 优先使用 preprocessing_stats.json 中的质量指标 (它们在去趋势前计算，更准确)
        stats = {}
        if stats_file.exists():
            with open(stats_file, "r") as f:
                stats = json.load(f)

        report_lines.append("### 3.1 信号质量")
        report_lines.append("")
        # 信号指标优先用 stats，避免去趋势后数据导致 SNR=0
        snr_mean = stats.get("mean_signal", metrics.get("signal_mean", 0))
        snr_std = stats.get("std_signal", metrics.get("signal_std", 0))
        snr_val = stats.get("snr", metrics.get("snr", 0))
        tsnr_mean = stats.get("tsnr_mean", metrics.get("tsnr_mean", 0))
        tsnr_med = stats.get("tsnr_median", metrics.get("tsnr_median", 0))
        report_lines.append(f"- 信号均值 (脑内体素): {snr_mean:.2f}")
        report_lines.append(f"- 信号标准差: {snr_std:.2f}")
        report_lines.append(f"- 信噪比 (SNR = mean/std): {snr_val:.2f}")
        report_lines.append(f"- 时间信噪比 (tSNR mean): {tsnr_mean:.2f}")
        report_lines.append(f"- 时间信噪比 (tSNR median): {tsnr_med:.2f}")
        report_lines.append(f"- 信号范围: [{metrics.get('min_signal', 0):.2f}, {metrics.get('max_signal', 0):.2f}]")

        report_lines.append("")
        report_lines.append("### 3.2 头动评估")
        report_lines.append("")

        # 头动指标优先用 stats 中的
        mean_fd_report = stats.get("mean_fd", metrics.get("mean_fd", 0))
        median_fd_report = stats.get("median_fd", metrics.get("median_fd", 0))
        max_fd_report = stats.get("max_fd", metrics.get("max_fd", 0))
        fd_05 = stats.get("fd_above_0.5mm", metrics.get("fd_above_0.5", 0))
        fd_02 = stats.get("fd_above_0.2mm", metrics.get("fd_above_0.2", 0))

        report_lines.append(f"- 平均 FD (帧位移): {mean_fd_report:.4f} mm")
        report_lines.append(f"- 中位 FD: {median_fd_report:.4f} mm")
        report_lines.append(f"- 最大 FD: {max_fd_report:.4f} mm")
        report_lines.append(f"- FD > 0.2mm 的时间点数: {fd_02}")
        report_lines.append(f"- FD > 0.5mm 的时间点数: {fd_05}")

        # 质量评级
        if mean_fd_report < 0.1:
            rating = "**优秀** - 头动极小 (< 0.1mm)，数据质量高"
        elif mean_fd_report < 0.2:
            rating = "**良好** - 头动在可接受范围 (< 0.2mm)"
        elif mean_fd_report < 0.5:
            rating = "**可接受** - 头动较大 (< 0.5mm)，建议进一步处理"
        else:
            rating = "**较差** - 头动严重 (≥ 0.5mm)，可能影响分析结果"

        report_lines.append(f"\n**头动质量评级: {rating}**")

        report_lines.append("")
        report_lines.append("### 3.3 时间序列特征")
        report_lines.append("")

        # 信号漂移
        if data_4d is not None:
            n_tp = data_4d.shape[3]
            signals = data_4d.reshape(-1, n_tp)
            time_std = np.std(signals, axis=0)
            drift_pct = abs(time_std[-1] - time_std[0]) / np.mean(time_std) * 100 if np.mean(time_std) > 0 else 0
            report_lines.append(f"- 信号时序漂移: {drift_pct:.2f}%")
            report_lines.append(f"- tSNR 中位数: {tsnr_med:.2f}")

    report_lines.append("")
    report_lines.append("## 4. 输出文件列表")
    report_lines.append("")

    # 列出输出文件
    for subdir in ["output", "output_fsl", "output_report"]:
        output_path = Path(fmri_data_dir).parents[0] / subdir
        if output_path.exists():
            report_lines.append(f"### {subdir}/")
            for f in sorted(output_path.rglob("*")):
                if f.is_file():
                    report_lines.append(f"- `{f.relative_to(Path(fmri_data_dir).parents[0])}` ({f.stat().st_size / 1024:.1f} KB)")
            report_lines.append("")

    report_lines.append("## 5. 建议")
    report_lines.append("")
    report_lines.append("- 如果使用 FSL FEAT，可将预处理后的数据直接输入 GLM 分析")
    report_lines.append("- 如果需要更高级的预处理，推荐使用 fMRIPrep")
    report_lines.append("- 运动参数 (motion_params.txt) 可作为 nuisance regressors 用于 GLM 分析")
    report_lines.append("- 建议设置 FD > 0.5mm 的 volumes 作为 censoring regressors")

    # 写入报告
    report_text = "\n".join(report_lines)
    report_path = output_dir / "fmri_analysis_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"  报告已保存: {report_path}")

    # 生成可视化
    if data_4d is not None:
        print("\n[分析] 生成可视化...")
        spatial_maps(data_4d, output_dir)
        temporal_analysis(data_4d, tr=2.0, output_dir=output_dir)

    return metrics if data_4d is not None else {}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="fMRI 分析与报告生成")
    parser.add_argument("--fmri-data-dir", type=str, default=None)
    parser.add_argument("--preprocess-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    fmri_root = Path(__file__).parents[2]
    fmri_data_dir = fmri_root / "output" / "nifti_fmri"
    preprocess_dir = fmri_root / "output" / "output_fsl"
    output_dir = fmri_root / "output" / "output_report"

    if args.fmri_data_dir:
        fmri_data_dir = Path(args.fmri_data_dir)
    if args.preprocess_dir:
        preprocess_dir = Path(args.preprocess_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)

    metrics = generate_report(fmri_data_dir, preprocess_dir, output_dir)
