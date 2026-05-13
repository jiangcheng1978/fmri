#!/usr/bin/env python3
"""
fMRI 数据预处理管线
包含: motion correction, 空间平滑, 去趋势, 去噪, 滤波
基于 Nibabel + Scipy 实现 (FFT 加速)
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import nibabel as nib

def load_nifti(filepath):
    """加载 NIfTI 文件"""
    img = nib.load(filepath)
    data = img.get_fdata()
    return img, data


def save_nifti(filepath, data, header_img):
    """保存 NIfTI 文件"""
    new_img = nib.Nifti1Image(data.astype(np.float32), header_img.affine, header_img.header)
    nib.save(new_img, filepath)
    print(f"  保存: {filepath} (shape={data.shape})")


def motion_correction_fft(data_4d, iterations=2):
    """
    FFT-based phase correlation motion correction
    将每个 volume 配准到平均 volume，快速高效
    使用 FFT 平移估计 + 粗搜+精搜两阶段匹配
    """
    from scipy.ndimage import zoom, affine_transform

    n_volumes = data_4d.shape[3]

    # 1) 平均 volume 作为参考 (比第一个更稳健)
    ref_volume = np.mean(data_4d, axis=3)

    # 2) 生成脑掩码
    mask = ref_volume > np.percentile(ref_volume, 5)

    ref_masked = ref_volume.copy()
    ref_masked[~mask] = 0

    aligned_data = np.zeros_like(data_4d)
    aligned_data[:, :, :, 0] = ref_masked.copy()
    motions = np.zeros((n_volumes, 6))  # 3 translation (mm) + 3 rotation (rad)

    # 3) 粗参考 = 前几个 volumes 的平均，避免运动偏置
    n_ref = min(5, n_volumes)
    coarse_ref = np.mean(data_4d[:, :, :, :n_ref], axis=3)
    coarse_ref_masked = coarse_ref.copy()
    coarse_ref_masked[~mask] = 0

    for t in range(1, n_volumes):
        vol = data_4d[:, :, :, t].copy()
        vol_masked = vol.copy()
        vol_masked[~mask] = 0

        # --- 第一阶段: FFT 粗估计 (亚体素精度) ---
        # 降采样到 22x22x22 (scale=0.25)
        ref_small = zoom(coarse_ref_masked, 0.25, order=1)
        vol_small = zoom(vol_masked, 0.25, order=1)
        ref_small = (ref_small - np.mean(ref_small)) / (np.std(ref_small) + 1e-10)
        vol_small = (vol_small - np.mean(vol_small)) / (np.std(vol_small) + 1e-10)

        from scipy.signal import fftconvolve
        corr = fftconvolve(ref_small, vol_small[::-1, ::-1, ::-1], mode='full')
        center = tuple(s // 2 for s in corr.shape)

        # FFT 峰值 (亚体素)
        peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
        peak_dx_ff = peak_idx[0] - center[0]
        peak_dy_ff = peak_idx[1] - center[1]
        peak_dz_ff = peak_idx[2] - center[2]

        # --- 第二阶段: 小范围精细搜索 (±1 体素) ---
        fine_range = 1
        best_score = -np.inf
        best_dx, best_dy, best_dz = 0, 0, 0

        for dx in range(-fine_range, fine_range + 1):
            for dy in range(-fine_range, fine_range + 1):
                for dz in range(-fine_range, fine_range + 1):
                    dd = [dx, dy, dz]
                    cx = (int(peak_dx_ff) + dd[0])
                    cy = (int(peak_dy_ff) + dd[1])
                    cz = (int(peak_dz_ff) + dd[2])
                    if 0 <= cx < corr.shape[0] and 0 <= cy < corr.shape[1] and 0 <= cz < corr.shape[2]:
                        score = corr[cx, cy, cz]
                        if score > best_score:
                            best_score = score
                            best_dx, best_dy, best_dz = peak_dx_ff + dd[0], peak_dy_ff + dd[1], peak_dz_ff + dd[2]

        # 恢复真实体素偏移
        real_dx = best_dx / 0.25  # / scale
        real_dy = best_dy / 0.25
        real_dz = best_dz / 0.25

        # 如果偏移可忽略，直接复制
        if abs(real_dx) < 0.05 and abs(real_dy) < 0.05 and abs(real_dz) < 0.05:
            aligned_data[:, :, :, t] = vol_masked.copy()
        else:
            shift_matrix = np.eye(4)
            shift_matrix[0, 3] = -real_dx
            shift_matrix[1, 3] = -real_dy
            shift_matrix[2, 3] = -real_dz
            aligned_vol = affine_transform(vol_masked, shift_matrix,
                                            output_shape=vol_masked.shape,
                                            mode='nearest', cval=0.0)
            aligned_data[:, :, :, t] = aligned_vol

        # 运动参数: 平移 (mm)
        motions[t] = [real_dx * 2.5, real_dy * 2.5, real_dz * 2.5, 0, 0, 0]

    return aligned_data, motions


def spatial_smooth_gaussian(data, fwhm=4.0):
    """
    空间高斯平滑 (使用 FFT 卷积加速)
    FWHM: 半高全宽 (mm)
    """
    from scipy.ndimage import gaussian_filter

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2))) / 2.5  # 转换为体素单位

    smoothed = np.zeros_like(data)
    for t in range(data.shape[3]):
        smoothed[:, :, :, t] = gaussian_filter(data[:, :, :, t], sigma=sigma)

    return smoothed


def detrend_signal(data_4d):
    """
    线性去趋势 (4D: x, y, z, time)
    对每个体素的时间序列减去均值和线性趋势
    """
    n_tp = data_4d.shape[3]
    t = np.arange(n_tp, dtype=np.float64)
    t_mean = np.mean(t)
    t_var = np.var(t)

    # 减去均值
    ts = data_4d - np.mean(data_4d, axis=-1, keepdims=True)

    if t_var > 0:
        # 对每个体素计算线性斜率
        slope = np.sum(ts * t[None, None, None, :], axis=-1) / t_var
        trend = slope[..., None] * (t - t_mean)
        ts -= trend

    return ts


def bandpass_filter(data, tr=2.0, low_cut=0.01, high_cut=0.1):
    """
    带通滤波 (FFT 方法)
    """
    n_timepoints = data.shape[3]
    nyquist = 1.0 / (2 * tr)
    low = low_cut / nyquist
    high = high_cut / nyquist

    # FFT 变换
    fft_data = np.fft.rfft(data, axis=3)
    freqs = np.fft.rfftfreq(n_timepoints, d=tr)

    # 构建带通掩码
    mask = np.ones(len(freqs))
    mask[np.abs(freqs) < low] = 0
    mask[np.abs(freqs) > high] = 0
    mask[0] = 1  # 保留直流分量

    fft_data *= mask[None, None, None, :]
    return np.fft.irfft(fft_data, n=n_timepoints, axis=3)


def compute_fd(motions):
    """
    计算 framewise displacement (瞬时位移)
    """
    if len(motions) < 2:
        return np.array([0])

    fd = np.zeros(len(motions))
    for i in range(1, len(motions)):
        dd = np.abs(motions[i, :3] - motions[i-1, :3])
        fd[i] = 0.5 * np.sum(dd)

    return fd


def preprocess_fmri(nii_files, output_dir, tr=2.0, fwhm=4.0, low_cut=0.01, high_cut=0.1):
    """
    完整 fMRI 预处理流程
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"fMRI 预处理管线")
    print(f"{'='*60}")

    # 1. 加载数据
    print("\n[Step 1] 加载数据...")
    nii_path = nii_files[0]
    img, data_4d = load_nifti(nii_path)

    n_volumes = data_4d.shape[3]
    n_slices = data_4d.shape[2]
    shape = data_4d.shape[:3]

    print(f"  数据维度: {data_4d.shape} (x, y, z, time)")
    print(f"  TR: {tr}s, 总时长: {n_volumes * tr}s")
    print(f"  体素: {shape[0]}x{shape[1]}x{shape[2]}")
    print(f"  Slice 数: {n_slices}")
    print(f"  信号范围: [{data_4d.min():.1f}, {data_4d.max():.1f}]")

    # 2. 头动校正
    print("\n[Step 2] 头动校正 (FFT-based)...")
    aligned_data, motions = motion_correction_fft(data_4d)
    fd = compute_fd(motions)
    max_fd = np.max(fd)
    mean_fd = np.mean(fd)
    print(f"  最大瞬时位移 (FD): {max_fd:.4f} mm")
    print(f"  平均 FD: {mean_fd:.4f} mm")
    print(f"  FD > 0.5mm: {np.sum(fd > 0.5)} volumes")
    print(f"  FD > 0.2mm: {np.sum(fd > 0.2)} volumes")

    # 3. 空间平滑
    print(f"\n[Step 3] 空间平滑 (FWHM={fwhm:.0f}mm)...")
    smoothed_data = spatial_smooth_gaussian(aligned_data, fwhm=fwhm)
    print("  完成")

    # 4. 去趋势
    print("\n[Step 4] 去趋势...")
    detrended_data = detrend_signal(smoothed_data)
    print("  完成线性去趋势")

    # 4.5 计算质量指标 (在去趋势之前，否则 tSNR = 0)
    print("\n[Step 4.5] 计算质量指标...")
    # tSNR: 每个体素的均值/标准差 (沿时间维度)
    tsnr_vol = np.mean(smoothed_data, axis=3) / (np.std(smoothed_data, axis=3) + 1e-10)
    tsnr_brain = tsnr_vol[tsnr_vol > 0]
    if len(tsnr_brain) > 0:
        tsnr_mean = float(np.mean(tsnr_brain))
        tsnr_median = float(np.median(tsnr_brain))
    else:
        tsnr_mean = tsnr_median = 0.0
    print(f"  tSNR 均值: {tsnr_mean:.1f}")
    print(f"  tSNR 中位数: {tsnr_median:.1f}")

    # 信号 SNR (基于所有体素)
    all_vals = smoothed_data[smoothed_data > np.percentile(smoothed_data, 5)]
    if len(all_vals) > 0:
        preproc_mean = float(np.mean(all_vals))
        preproc_std = float(np.std(all_vals))
        preproc_snr = preproc_mean / preproc_std if preproc_std > 0 else 0.0
    else:
        preproc_mean = float(np.mean(smoothed_data))
        preproc_std = float(np.std(smoothed_data))
        preproc_snr = preproc_mean / preproc_std if preproc_std > 0 else 0.0

    # 5. 带通滤波
    print(f"\n[Step 5] 带通滤波 ({low_cut}-{high_cut} Hz)...")
    filtered_data = bandpass_filter(detrended_data, tr=tr, low_cut=low_cut, high_cut=high_cut)
    print("  完成滤波")

    # 6. 保存预处理后的数据
    print("\n[Step 6] 保存结果...")
    save_nifti(output_dir / "sub-003_preproc.nii.gz", filtered_data, img)

    # 7. 保存运动参数
    np.savetxt(output_dir / "motion_params.txt", motions,
               header="tx(mm) ty(mm) tz(mm) rx(rad) ry(rad) rz(rad) per volume")

    # 8. 保存 FD
    np.savetxt(output_dir / "framewise_displacement.txt", fd,
               header="FD per volume (mm)")

    # 9. 保存统计
    stats = {
        "n_volumes": int(n_volumes),
        "n_slices": int(n_slices),
        "tr": float(tr),
        "total_duration": float(n_volumes * tr),
        "shape": [int(s) for s in shape],
        "max_fd": float(max_fd),
        "mean_fd": float(mean_fd),
        "median_fd": float(np.median(fd)),
        "fd_above_0.5mm": int(np.sum(fd > 0.5)),
        "fd_above_0.2mm": int(np.sum(fd > 0.2)),
        "mean_signal": float(preproc_mean),
        "std_signal": float(preproc_std),
        "snr": float(preproc_snr),
        "tsnr_mean": float(tsnr_mean),
        "tsnr_median": float(tsnr_median),
        "max_trans_mm": float(np.max(np.abs(motions[:, :3]))),
        "fwhm_smooth_mm": float(fwhm),
        "bandpass_hz": [float(low_cut), float(high_cut)],
        "timestamp": str(datetime.now()),
    }

    with open(output_dir / "preprocessing_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[INFO] 预处理完成!")
    print(f"  输出目录: {output_dir}")
    print(f"  预处理数据: sub-003_preproc.nii.gz")
    print(f"  运动参数: motion_params.txt")
    print(f"  帧位移: framewise_displacement.txt")
    print(f"  统计信息: preprocessing_stats.json")

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="fMRI 预处理")
    parser.add_argument("--input", type=str, default=None,
                        help="fMRI NIfTI 输入路径")
    parser.add_argument("--output", type=str, default=None,
                        help="预处理输出目录")
    parser.add_argument("--tr", type=float, default=2.0, help="重复时间 (秒)")
    parser.add_argument("--fwhm", type=float, default=4.0, help="空间平滑 FWHM (mm)")
    parser.add_argument("--low-cut", type=float, default=0.01, help="高通截止 (Hz)")
    parser.add_argument("--high-cut", type=float, default=0.1, help="低通截止 (Hz)")
    args = parser.parse_args()

    fmri_root = Path(__file__).parents[2]
    data_dir = fmri_root / "output" / "nifti_fmri"

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = data_dir / "sub-003_14.nii.gz"

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = fmri_root / "output" / "output_fsl"

    if not input_path.exists():
        print(f"[ERROR] 未找到 fMRI 数据: {input_path}")
        print(f"[INFO] 请先运行: python scripts/dicom/dicom_convert.py")
        sys.exit(1)

    stats = preprocess_fmri([input_path], output_dir, tr=args.tr,
                           fwhm=args.fwhm,
                           low_cut=args.low_cut,
                           high_cut=args.high_cut)
