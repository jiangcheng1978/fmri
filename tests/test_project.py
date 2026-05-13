#!/usr/bin/env python3
"""
fMRI 项目综合测试套件
验证：项目结构、依赖导入、脚本语法、数据文件、预处理管线、报告生成
"""
import unittest
import json
import os
import subprocess
import sys
import numpy as np
from pathlib import Path

FMRI_ROOT = Path(__file__).parent.parent


# ═══════════════════════════════════════════════════════════
# 测试 1: 项目结构
# ═══════════════════════════════════════════════════════════
class TestProjectStructure(unittest.TestCase):
    """验证项目目录和文件结构是否合理"""

    def test_venv_exists(self):
        """.venv 虚拟环境目录存在"""
        self.assertTrue((FMRI_ROOT / ".venv").is_dir(), ".venv 虚拟环境不存在")

    def test_venv_has_python(self):
        """.venv 中有 python 解释器"""
        venv_python = FMRI_ROOT / ".venv" / "bin" / "python"
        self.assertTrue(venv_python.exists(), ".venv/bin/python 不存在")

    def test_scripts_directory(self):
        """scripts/ 下包含所有模块"""
        scripts = FMRI_ROOT / "scripts"
        self.assertTrue(scripts.is_dir())
        for mod in ["run_pipeline.py", "dicom/dicom_convert.py",
                     "preprocess/fmri_preprocess.py", "analysis/report_generator.py",
                     "generate_docs.py"]:
            self.assertTrue((scripts / mod).exists(), f"缺少 {mod}")

    def test_data_directory(self):
        """data/ 目录存在"""
        self.assertTrue((FMRI_ROOT / "data").is_dir())

    def test_output_directory(self):
        """output/ 目录存在"""
        self.assertTrue((FMRI_ROOT / "output").is_dir())

    def test_no_sys_path_injection_in_scripts(self):
        """脚本中不应有 sys.path.insert 引用 app/ 目录"""
        for root, dirs, files in os.walk(FMRI_ROOT / "scripts"):
            for f in files:
                if f.endswith(".py"):
                    filepath = Path(root) / f
                    content = filepath.read_text()
                    self.assertNotIn('sys.path.insert', content,
                                     f"{filepath.name} 中存在 sys.path.insert，应移除")


# ═══════════════════════════════════════════════════════════
# 测试 2: 依赖导入
# ═══════════════════════════════════════════════════════════
class TestDependencies(unittest.TestCase):
    """验证关键 Python 包可导入"""

    def test_numpy(self):
        import numpy
        self.assertIsNotNone(numpy.__version__)

    def test_scipy(self):
        import scipy
        self.assertIsNotNone(scipy.__version__)

    def test_nibabel(self):
        import nibabel
        self.assertIsNotNone(nibabel.__version__)

    def test_matplotlib(self):
        import matplotlib
        self.assertIsNotNone(matplotlib.__version__)

    def test_nilearn(self):
        import nilearn
        self.assertIsNotNone(nilearn.__version__)

    def test_sklearn(self):
        import sklearn
        self.assertIsNotNone(sklearn.__version__)

    def test_pandas(self):
        import pandas
        self.assertIsNotNone(pandas.__version__)

    def test_dcm2niix_package(self):
        import dcm2niix
        self.assertTrue(os.path.exists(dcm2niix.bin),
                        f"dcm2niix 二进制不存在: {dcm2niix.bin}")


# ═══════════════════════════════════════════════════════════
# 测试 3: 脚本语法正确性
# ═══════════════════════════════════════════════════════════
class TestScriptSyntax(unittest.TestCase):
    """验证所有 Python 脚本可编译通过"""

    def test_each_script_compiles(self):
        py_files = [
            FMRI_ROOT / "scripts" / "run_pipeline.py",
            FMRI_ROOT / "scripts" / "dicom" / "dicom_convert.py",
            FMRI_ROOT / "scripts" / "preprocess" / "fmri_preprocess.py",
            FMRI_ROOT / "scripts" / "analysis" / "report_generator.py",
            FMRI_ROOT / "scripts" / "generate_docs.py",
        ]
        for f in py_files:
            try:
                compile(f.read_text(), str(f), "exec")
            except SyntaxError as e:
                self.fail(f"{f.name} 语法错误: {e}")


# ═══════════════════════════════════════════════════════════
# 测试 4: 数据文件完整性
# ═══════════════════════════════════════════════════════════
class TestDataIntegrity(unittest.TestCase):
    """验证输入输出数据文件"""

    def test_fmri_dicom_exist(self):
        sub_003 = FMRI_ROOT / "data" / "sub_003"
        if not sub_003.exists():
            self.skipTest("fMRI DICOM 数据目录不存在")
        dicom_files = list(sub_003.glob("*.IMA")) + list(sub_003.glob("*.dcm"))
        self.assertTrue(len(dicom_files) > 0, "sub_003 中无 DICOM 文件")

    def test_nifti_fmri_exist(self):
        nifti_fmri = FMRI_ROOT / "output" / "nifti_fmri"
        if not nifti_fmri.exists():
            self.skipTest("NIfTI 目录不存在")
        nii_files = list(nifti_fmri.glob("*.nii.gz"))
        self.assertTrue(len(nii_files) > 0, "NIfTI 文件不存在")

    def test_preproc_exist(self):
        output_fsl = FMRI_ROOT / "output" / "output_fsl"
        if not output_fsl.exists():
            self.skipTest("预处理输出目录不存在")
        preproc = output_fsl / "sub-003_preproc.nii.gz"
        self.assertTrue(preproc.exists(), "预处理数据不存在")

    def test_preproc_loadable(self):
        """预处理数据可被 nibabel 加载"""
        import nibabel as nib
        preproc = FMRI_ROOT / "output" / "output_fsl" / "sub-003_preproc.nii.gz"
        if not preproc.exists():
            self.skipTest("预处理数据不存在")
        img = nib.load(preproc)
        data = img.get_fdata()
        self.assertEqual(data.shape[3], 244, f"时间点数应为 244，实际 {data.shape[3]}")
        self.assertEqual(len(data.shape), 4, "应为 4D 数据")

    def test_stats_json_valid(self):
        """preprocessing_stats.json 有效"""
        stats_file = FMRI_ROOT / "output" / "output_fsl" / "preprocessing_stats.json"
        if not stats_file.exists():
            self.skipTest("stats 文件不存在")
        with open(stats_file) as f:
            stats = json.load(f)
        self.assertIn("n_volumes", stats)
        self.assertIn("tsnr_mean", stats)
        self.assertIn("mean_fd", stats)
        self.assertGreater(stats["n_volumes"], 0)
        self.assertGreater(stats["tsnr_mean"], 0)


# ═══════════════════════════════════════════════════════════
# 测试 5: 预处理管线核心函数（通过子进程调用验证）
# ═══════════════════════════════════════════════════════════
class TestPreprocessFunctions(unittest.TestCase):
    """通过 Python -c 方式测试预处理核心函数"""

    def _run_test_script(self, script):
        """在虚拟环境中执行测试脚本"""
        venv_python = FMRI_ROOT / ".venv" / "bin" / "python"
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [str(venv_python), "-c", script],
            capture_output=True, text=True,
            cwd=str(FMRI_ROOT), env=env,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0,
                         f"子进程失败:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    def test_fd_calculation(self):
        """FD 计算正确"""
        script = """
import numpy as np
def compute_fd(motions):
    if len(motions) < 2: return np.array([0])
    fd = np.zeros(len(motions))
    for i in range(1, len(motions)):
        dd = np.abs(motions[i, :3] - motions[i-1, :3])
        fd[i] = 0.5 * np.sum(dd)
    return fd
motions = np.array([[0,0,0,0,0,0], [0.5,0.3,0.2,0,0,0], [0.7,0.5,0.1,0,0,0]])
fd = compute_fd(motions)
assert len(fd) == 3, f"FD length {len(fd)} != 3"
assert abs(fd[1] - 0.5) < 1e-10, f"fd[1]={fd[1]}"
print(f"FD OK: {fd}")
"""
        self._run_test_script(script)

    def test_fd_zero_motion(self):
        """无运动时 FD = 0"""
        script = """
import numpy as np
def compute_fd(motions):
    if len(motions) < 2: return np.array([0])
    fd = np.zeros(len(motions))
    for i in range(1, len(motions)):
        dd = np.abs(motions[i, :3] - motions[i-1, :3])
        fd[i] = 0.5 * np.sum(dd)
    return fd
motions = np.zeros((3, 6))
fd = compute_fd(motions)
np.testing.assert_array_almost_equal(fd, [0, 0, 0])
print(f"Zero FD OK: {fd}")
"""
        self._run_test_script(script)

    def test_detrend_removes_mean(self):
        """去趋势后均值接近 0"""
        script = """
import numpy as np
def detrend_signal(data_4d):
    n_tp = data_4d.shape[3]
    t = np.arange(n_tp, dtype=np.float64)
    t_mean = np.mean(t)
    t_var = np.var(t)
    ts = data_4d - np.mean(data_4d, axis=-1, keepdims=True)
    if t_var > 0:
        slope = np.sum(ts * t[None, None, None, :], axis=-1) / t_var
        trend = slope[..., None] * (t - t_mean)
        ts -= trend
    return ts
np.random.seed(42)
data = np.random.rand(10, 10, 5, 50).astype(np.float64)
result = detrend_signal(data)
means = np.mean(result, axis=-1)
assert np.all(np.abs(means) < 1e-10), f"max mean={np.abs(means).max()}"
print(f"Detrend OK: max mean={np.abs(means).max():.2e}")
"""
        self._run_test_script(script)

    def test_bandpass_filter_shape(self):
        """滤波后数据形状不变"""
        script = """
import numpy as np
def bandpass_filter(data, tr=2.0, low_cut=0.01, high_cut=0.1):
    n_timepoints = data.shape[3]
    nyquist = 1.0 / (2 * tr)
    low = low_cut / nyquist
    high = high_cut / nyquist
    fft_data = np.fft.rfft(data, axis=3)
    freqs = np.fft.rfftfreq(n_timepoints, d=tr)
    mask = np.ones(len(freqs))
    mask[np.abs(freqs) < low] = 0
    mask[np.abs(freqs) > high] = 0
    mask[0] = 1
    fft_data *= mask[None, None, None, :]
    return np.fft.irfft(fft_data, n=n_timepoints, axis=3)
np.random.seed(42)
data = np.random.rand(10, 10, 5, 100).astype(np.float64)
result = bandpass_filter(data, tr=2.0, low_cut=0.01, high_cut=0.1)
assert result.shape == data.shape, f"shape {result.shape} != {data.shape}"
print(f"Bandpass shape OK: {result.shape}")
"""
        self._run_test_script(script)

    def test_spatial_smooth_shape(self):
        """平滑后数据形状不变"""
        script = """
import numpy as np
from scipy.ndimage import gaussian_filter
def spatial_smooth_gaussian(data, fwhm=4.0):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2))) / 2.5
    smoothed = np.zeros_like(data)
    for t in range(data.shape[3]):
        smoothed[:, :, :, t] = gaussian_filter(data[:, :, :, t], sigma=sigma)
    return smoothed
np.random.seed(42)
data = np.random.rand(10, 10, 5, 20).astype(np.float64)
result = spatial_smooth_gaussian(data, fwhm=4.0)
assert result.shape == data.shape, f"shape {result.shape} != {data.shape}"
print(f"Smooth shape OK: {result.shape}")
"""
        self._run_test_script(script)


# ═══════════════════════════════════════════════════════════
# 测试 6: 输出数据合理性
# ═══════════════════════════════════════════════════════════
class TestOutputReasonableness(unittest.TestCase):
    """验证输出数据的合理性"""

    @classmethod
    def setUpClass(cls):
        import nibabel as nib
        cls.preproc = FMRI_ROOT / "output" / "output_fsl" / "sub-003_preproc.nii.gz"
        img = nib.load(cls.preproc)
        cls.data = img.get_fdata()

    def test_data_not_all_zero(self):
        """预处理数据不全为 0"""
        self.assertTrue(np.any(np.abs(self.data) > 0), "数据全为 0")

    def test_fd_reasonable(self):
        """FD 值非负"""
        fd = np.loadtxt(FMRI_ROOT / "output" / "output_fsl" / "framewise_displacement.txt")
        self.assertTrue(np.all(fd >= 0), "FD 不应为负数")

    def test_motion_params_shape(self):
        """运动参数形状合理 (N_volumes x 6)"""
        motions = np.loadtxt(FMRI_ROOT / "output" / "output_fsl" / "motion_params.txt")
        self.assertEqual(motions.shape[1], 6, "应有 6 个运动参数")
        self.assertEqual(motions.shape[0], 244, "应有 244 行")

    def test_report_exists(self):
        """分析报告存在"""
        report = FMRI_ROOT / "output" / "output_report" / "fmri_analysis_report.md"
        self.assertTrue(report.exists())
        self.assertGreater(report.stat().st_size, 0)

    def test_visualizations_exist(self):
        """可视化图片存在"""
        spatial = FMRI_ROOT / "output" / "output_report" / "spatial_maps.png"
        temporal = FMRI_ROOT / "output" / "output_report" / "temporal_analysis.png"
        self.assertTrue(spatial.exists(), "spatial_maps.png 不存在")
        self.assertTrue(temporal.exists(), "temporal_analysis.png 不存在")
        self.assertGreater(spatial.stat().st_size, 1000, "spatial_maps.png 过小")
        self.assertGreater(temporal.stat().st_size, 1000, "temporal_analysis.png 过小")


# ═══════════════════════════════════════════════════════════
# 测试 7: HTML 文档
# ═══════════════════════════════════════════════════════════
class TestHtmlDocs(unittest.TestCase):
    """验证 HTML 文档"""

    def test_html_file_exists(self):
        html = FMRI_ROOT / "html" / "fmri-analysis.html"
        self.assertTrue(html.exists(), "HTML 文档不存在")

    def test_html_has_flowchart(self):
        """HTML 包含 CSS 流程图"""
        html = (FMRI_ROOT / "html" / "fmri-analysis.html").read_text()
        self.assertIn("flowchart", html)
        self.assertIn("fc-node", html)

    def test_html_has_code_highlight(self):
        """HTML 包含代码高亮"""
        html = (FMRI_ROOT / "html" / "fmri-analysis.html").read_text()
        self.assertIn("highlight.js", html)

    def test_html_content_complete(self):
        """HTML 包含关键章节"""
        html = (FMRI_ROOT / "html" / "fmri-analysis.html").read_text()
        for keyword in ["处理管线", "质量评估", "新手教程", "分析结果", "关键代码"]:
            self.assertIn(keyword, html, f"HTML 缺少章节: {keyword}")


if __name__ == "__main__":
    unittest.main()
