#!/usr/bin/env python3
"""Install the executable Nardon convention lock in the public RMP tutorial."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


LANGUAGES = ("en", "zh", "ja", "ko", "de", "fr", "ru")
THEORY_TAG = "nardon-convention-theory"
EXECUTABLE_TAG = "nardon-convention-executable"


ENGLISH_MARKDOWN = (
    r"""## [NARDON_CONVENTION] Convention lock: Nardon's Eqs. (3.3)-(3.17)

This section fixes the notation used by the executable spectrum API. It follows Eric Nardon's 2007 thesis, Chapter 3 and Appendix A, rather than inferring signs from a plot. The public source is [HAL pastel-00003137](https://pastel.hal.science/pastel-00003137v1).

On healed magnetic surfaces use \((s,\theta^*,\varphi)\), with \(s=\sqrt{\psi}\) and
\[
\frac{\mathrm d\varphi}{\mathrm d\theta^*}=q(s).
\]
The contravariant components are
\[
B^1=\mathbf B\cdot\nabla s,\qquad
B^2=\mathbf B\cdot\nabla\theta^*,\qquad
B^3=\mathbf B\cdot\nabla\varphi.
\]
Two radial quantities in the thesis must not be conflated:
\[
b^r=\frac{B^1/B_{\rm axis}}{\sqrt{g^{11}}},
\qquad
\widetilde b^1=\frac{B^1}{B^3}.
\]
Nardon's island-width derivation below uses the Fourier coefficients of \(\widetilde b^1\), evaluated from the perturbation \(\delta\mathbf B\) on coordinates healed from the integrable background \(\mathbf B_0\).""",
    r"""### Signed Fourier indices and the reality condition

We name the thesis toroidal index \(n_N\) to distinguish it from a positive resonance-family label \(n_0\):
\[
\widetilde b^1_{m n_N}(s)
=\int_0^{2\pi}\!\!\int_0^{2\pi}
\widetilde b^1(s,\theta^*,\varphi)
e^{-i(m\theta^*+n_N\varphi)}
\frac{\mathrm d\theta^*}{2\pi}\frac{\mathrm d\varphi}{2\pi},
\]
\[
\widetilde b^1
=\sum_{m,n_N}\widetilde b^1_{m n_N}
e^{i(m\theta^*+n_N\varphi)}.
\]
For a real field,
\[
\widetilde b^1_{-m,-n_N}
=\left(\widetilde b^1_{m n_N}\right)^*.
\]
This does **not** make \((m,n_N)\) and \((m,-n_N)\) conjugates. They are distinct helicity branches and may have different amplitudes and phases.

If a field is sampled over one of \(N_{\rm fp}\) identical field periods, local FFT harmonic \(k\) maps to the full-torus thesis index as
\[
n_N=N_{\rm fp}k.
\]
In pyna, use `nardon_n`, `nardon_mode_index`, and `nardon_mode_coefficient` for this signed index. `physical_n` remains a compatibility alias for the historical basis \(e^{i(m\theta^*-n\varphi)}\).""",
    r"""### Resonance selection, amplitude, width, and phase

Along an unperturbed field line, \(\varphi=q\theta^*+\mathrm{const}\). For positive \(q=m/n_0\), the stationary helical phase is \(m\theta^*-n_0\varphi\), so the resonant pair is
\[
(m,-n_0),\qquad(-m,+n_0),
\]
not \((m,+n_0)\). More generally pyna selects
\[
n_N=-\operatorname{sign}(q)n_0.
\]
Nardon's definitions are
\[
\widetilde b^1_{\rm res}=2\left|\widetilde b^1_{m,-n_0}\right|,
\qquad
\delta_{q=m/n_0}
=\left(\frac{4q^2\widetilde b^1_{\rm res}}{q'm}\right)^{1/2},
\qquad q'=\frac{\mathrm dq}{\mathrm ds}.
\]
Here \(\delta\) is the **half-width in the same \(s\) coordinate**. The plotted curved bar must use this physical width; zoom or sampling density may change, but the width must not be visually inflated.

For adjacent \(q=m/n_0\) and \(q=(m+1)/n_0\) surfaces, Eq. (3.17) is
\[
\sigma_{\rm Chir}^{m,m+1}
=\frac{\delta_{q=m/n_0}+\delta_{q=(m+1)/n_0}}
{\Delta_{m,m+1}},
\]
where \(\Delta_{m,m+1}\) is their separation measured in the same radial coordinate. Cross-family overlap generalizations must preserve that coordinate and half-width contract.

At fixed \(\varphi\), a coefficient-phase change \(\Delta\alpha\) moves a fixed helical phase by
\[
\Delta\theta^*=-\frac{\Delta\alpha}{m}.
\]
This relative shift is convention-stable. Assigning the absolute phase to an X point or O point additionally requires the sign of shear, field/angle orientation, and the fixed-point map convention, so it must be checked against Newton/cyna periodic points.""",
    r"""### Operational convention checklist

Before interpreting a spectrum or optimizing island topology:

1. Form an explicit, provenance-checked decomposition \(\mathbf B=\mathbf B_0+\delta\mathbf B\).
2. Heal closed nested surfaces and obtain \(q(s)\), \(q'(s)\), and straight-field-line \(\theta^*\) from \(\mathbf B_0\).
3. Project \(\delta\mathbf B\), not the total field, into \(\widetilde b^1=B^1/B^3\).
4. Address Fourier coefficients with signed Nardon indices \((m,n_N)\).
5. For positive \(q=m/n_0\), read \((m,-n_0)\); for negative \(q\), read \((m,+n_0)\).
6. Check conjugacy only between \((m,n_N)\) and \((-m,-n_N)\).
7. Validate predicted phase and half-width against nonlinear fixed points, Poincare sections, and manifolds.
8. Record the origins and orientations of \(\theta^*\) and \(\varphi\); phase values are meaningless without them.""",
)


CHINESE_MARKDOWN = (
    r"""## [NARDON_CONVENTION] 约定锁：Nardon 论文式 (3.3)-(3.17)

本节固定可执行磁谱 API 的符号，严格沿用 Eric Nardon 2007 年博士论文第 3 章和附录 A，而不是从图形外观猜测正负号。公开原文见 [HAL pastel-00003137](https://pastel.hal.science/pastel-00003137v1)。

在愈合磁面上采用 \((s,\theta^*,\varphi)\)，其中 \(s=\sqrt{\psi}\)，且
\[
\frac{\mathrm d\varphi}{\mathrm d\theta^*}=q(s).
\]
逆变分量定义为
\[
B^1=\mathbf B\cdot\nabla s,\qquad
B^2=\mathbf B\cdot\nabla\theta^*,\qquad
B^3=\mathbf B\cdot\nabla\varphi.
\]
论文中的两个径向量不能混用：
\[
b^r=\frac{B^1/B_{\rm axis}}{\sqrt{g^{11}}},
\qquad
\widetilde b^1=\frac{B^1}{B^3}.
\]
下面的 Nardon 岛宽推导使用 \(\widetilde b^1\) 的 Fourier 系数；它必须由可积背景 \(\mathbf B_0\) 愈合出的坐标面上采样扰动场 \(\delta\mathbf B\) 得到。""",
    r"""### 有符号 Fourier 指标与实场共轭关系

这里把论文中的环向指标记为 \(n_N\)，以区别正的共振族标签 \(n_0\)：
\[
\widetilde b^1_{m n_N}(s)
=\int_0^{2\pi}\!\!\int_0^{2\pi}
\widetilde b^1(s,\theta^*,\varphi)
e^{-i(m\theta^*+n_N\varphi)}
\frac{\mathrm d\theta^*}{2\pi}\frac{\mathrm d\varphi}{2\pi},
\]
\[
\widetilde b^1
=\sum_{m,n_N}\widetilde b^1_{m n_N}
e^{i(m\theta^*+n_N\varphi)}.
\]
实场只保证
\[
\widetilde b^1_{-m,-n_N}
=\left(\widetilde b^1_{m n_N}\right)^*.
\]
它绝不意味着 \((m,n_N)\) 与 \((m,-n_N)\) 共轭；这两支 helicity 可以具有完全不同的幅值和相位。

若只采样 \(N_{\rm fp}\) 个相同场周期中的一个，局部 FFT 谐波 \(k\) 与全环面论文指标满足
\[
n_N=N_{\rm fp}k.
\]
pyna 中应使用 `nardon_n`、`nardon_mode_index` 和 `nardon_mode_coefficient` 访问该有符号指标。`physical_n` 只保留为历史基底 \(e^{i(m\theta^*-n\varphi)}\) 的兼容别名。""",
    r"""### 共振支、幅值、岛宽与相位

沿未扰动磁力线有 \(\varphi=q\theta^*+\mathrm{const}\)。当正 \(q=m/n_0\) 时，静止的螺旋相位是 \(m\theta^*-n_0\varphi\)，所以共振共轭对为
\[
(m,-n_0),\qquad(-m,+n_0),
\]
而不是 \((m,+n_0)\)。一般情况下 pyna 选择
\[
n_N=-\operatorname{sign}(q)n_0.
\]
Nardon 的定义为
\[
\widetilde b^1_{\rm res}=2\left|\widetilde b^1_{m,-n_0}\right|,
\qquad
\delta_{q=m/n_0}
=\left(\frac{4q^2\widetilde b^1_{\rm res}}{q'm}\right)^{1/2},
\qquad q'=\frac{\mathrm dq}{\mathrm ds}.
\]
这里 \(\delta\) 是**同一个 \(s\) 坐标中的半宽**。曲线宽度条只能画这个物理宽度；可以放大视窗或提高采样分辨率，但不能人为放大宽度。

对相邻的 \(q=m/n_0\) 与 \(q=(m+1)/n_0\) 磁面，式 (3.17) 为
\[
\sigma_{\rm Chir}^{m,m+1}
=\frac{\delta_{q=m/n_0}+\delta_{q=(m+1)/n_0}}
{\Delta_{m,m+1}},
\]
其中 \(\Delta_{m,m+1}\) 必须是在同一个径向坐标中量出的磁面间距。扩展到不同环向模族时也必须保持同一坐标与半宽定义。

固定 \(\varphi\) 时，系数相位变化 \(\Delta\alpha\) 导致固定螺旋相位移动
\[
\Delta\theta^*=-\frac{\Delta\alpha}{m}.
\]
该相对位移不依赖任意相位零点。绝对 X/O 相位还取决于磁剪切符号、磁场和角度方向以及周期映射约定，必须用 Newton/cyna 周期点核验。""",
    r"""### 操作核查清单

解释磁谱或优化边界磁岛前必须逐项满足：

1. 显式构造并校验来源的 \(\mathbf B=\mathbf B_0+\delta\mathbf B\)。
2. 用 \(\mathbf B_0\) 愈合闭合嵌套磁面，并计算 \(q(s)\)、\(q'(s)\) 和直场线角 \(\theta^*\)。
3. 将 \(\delta\mathbf B\) 而不是总场投影到 \(\widetilde b^1=B^1/B^3\)。
4. 始终用有符号 Nardon 指标 \((m,n_N)\) 访问 Fourier 系数。
5. 正 \(q=m/n_0\) 读取 \((m,-n_0)\)，负 \(q\) 读取 \((m,+n_0)\)。
6. 只在 \((m,n_N)\) 与 \((-m,-n_N)\) 之间检查共轭关系。
7. 用非线性 fixed point、Poincare 截面和流形回验预测相位与半宽。
8. 记录 \(\theta^*\) 和 \(\varphi\) 的零点与方向；缺少这些信息的相位数值没有可比性。""",
)


CODE_CELLS = (
    r"""# Executable convention check: independent opposite-helicity branches.
import numpy as np

from pyna.toroidal.perturbation_spectrum import (
    analyze_resonant_island_chains,
    nardon_island_half_width,
    nardon_resonant_amplitude,
    radial_perturbation_Fourier_spectrum,
)

_twopi = 2.0 * np.pi
theta_nardon = 0.17 + np.arange(128) * (_twopi / 128)
phi_nardon = -0.23 + np.arange(96) * (_twopi / 96)
m_nardon, n0_nardon = 3, 2
c_res_nardon = 2.5e-4 * np.exp(0.63j)       # (m, -n0), resonant for q > 0
c_opp_nardon = 0.8e-4 * np.exp(-0.27j)     # (m, +n0), distinct helicity

phase_res_nardon = (
    m_nardon * theta_nardon[None, :]
    - n0_nardon * phi_nardon[:, None]
)
phase_opp_nardon = (
    m_nardon * theta_nardon[None, :]
    + n0_nardon * phi_nardon[:, None]
)
tilde_b1_nardon = 2.0 * np.real(
    c_res_nardon * np.exp(1j * phase_res_nardon)
    + c_opp_nardon * np.exp(1j * phase_opp_nardon)
)
spectrum_nardon = radial_perturbation_Fourier_spectrum(
    tilde_b1_nardon,
    theta_nardon,
    phi_nardon,
    m_max=4,
    n_max=3,
    min_amplitude=1.0e-14,
)

np.testing.assert_allclose(
    spectrum_nardon.nardon_mode_coefficient(m_nardon, -n0_nardon),
    c_res_nardon,
    atol=2.0e-13,
)
np.testing.assert_allclose(
    spectrum_nardon.nardon_mode_coefficient(-m_nardon, +n0_nardon),
    c_res_nardon.conjugate(),
    atol=2.0e-13,
)
np.testing.assert_allclose(
    spectrum_nardon.nardon_mode_coefficient(m_nardon, +n0_nardon),
    c_opp_nardon,
    atol=2.0e-13,
)
assert not np.isclose(c_res_nardon, c_opp_nardon)""",
    r"""# One-field-period sampling: local k maps to Nardon's global n_N=N_fp*k.
nfp_nardon = 5
phi_period_nardon = 0.11 + np.arange(80) * (_twopi / nfp_nardon / 80)
period_phase_nardon = (
    m_nardon * theta_nardon[None, :]
    - nfp_nardon * phi_period_nardon[:, None]
)
period_grid_nardon = 2.0 * np.real(
    c_res_nardon * np.exp(1j * period_phase_nardon)
)
period_spectrum_nardon = radial_perturbation_Fourier_spectrum(
    period_grid_nardon,
    theta_nardon,
    phi_period_nardon,
    nfp=nfp_nardon,
    m_max=m_nardon,
    n_max=nfp_nardon,
    min_amplitude=1.0e-14,
)
period_idx_nardon = period_spectrum_nardon.nardon_mode_index(
    m_nardon, -nfp_nardon
)
assert period_idx_nardon is not None
assert period_spectrum_nardon.field_period_harmonic[period_idx_nardon] == -1
assert period_spectrum_nardon.nardon_n[period_idx_nardon] == -nfp_nardon
np.testing.assert_allclose(
    period_spectrum_nardon.dBr[period_idx_nardon],
    c_res_nardon,
    atol=2.0e-13,
)""",
    r"""# Resonance, half-width, and phase-shift checks in Nardon's s=sqrt(psi).
s_nardon = np.linspace(0.4, 0.8, 9)
q_nardon = 1.2 + 0.5 * s_nardon  # q=3/2 at s=0.6, q'=0.5
radial_grid_nardon = (
    (1.0 + 0.2 * (s_nardon - 0.6))[:, None, None]
    * tilde_b1_nardon[None, :, :]
)
radial_spectrum_nardon = radial_perturbation_Fourier_spectrum(
    radial_grid_nardon,
    theta_nardon,
    phi_nardon,
    radial_labels=s_nardon,
    layout="radial-phi-theta",
    m_max=4,
    n_max=3,
    min_amplitude=1.0e-14,
)
chains_nardon = analyze_resonant_island_chains(
    radial_spectrum_nardon,
    q_nardon,
    n=n0_nardon,
    m_values=[m_nardon],
)
assert len(chains_nardon) == 1
chain_nardon = chains_nardon[0]
assert chain_nardon.coefficient_n == -n0_nardon
np.testing.assert_allclose(chain_nardon.radial_label, 0.6, atol=1.0e-13)

bres_nardon = nardon_resonant_amplitude(c_res_nardon)
expected_half_width_nardon = np.sqrt(
    4.0 * (m_nardon / n0_nardon) ** 2
    * bres_nardon
    / (0.5 * m_nardon)
)
np.testing.assert_allclose(
    nardon_island_half_width(
        m_nardon / n0_nardon,
        0.5,
        m_nardon,
        bres_nardon,
    ),
    expected_half_width_nardon,
)
np.testing.assert_allclose(chain_nardon.half_width, expected_half_width_nardon)

delta_alpha_nardon = 0.36
delta_theta_star_nardon = -delta_alpha_nardon / m_nardon
np.testing.assert_allclose(
    m_nardon * delta_theta_star_nardon + delta_alpha_nardon,
    0.0,
    atol=1.0e-15,
)""",
)


def _tagged(cell, tag: str):
    cell.metadata["tags"] = [tag]
    return cell


def convention_cells(*, chinese: bool = False) -> list:
    """Return fresh notebook cells for the convention-locked section."""

    markdown = CHINESE_MARKDOWN if chinese else ENGLISH_MARKDOWN
    return [
        _tagged(nbf.v4.new_markdown_cell(markdown[0]), THEORY_TAG),
        _tagged(nbf.v4.new_markdown_cell(markdown[1]), THEORY_TAG),
        _tagged(nbf.v4.new_code_cell(CODE_CELLS[0]), EXECUTABLE_TAG),
        _tagged(nbf.v4.new_code_cell(CODE_CELLS[1]), EXECUTABLE_TAG),
        _tagged(nbf.v4.new_markdown_cell(markdown[2]), THEORY_TAG),
        _tagged(nbf.v4.new_code_cell(CODE_CELLS[2]), EXECUTABLE_TAG),
        _tagged(nbf.v4.new_markdown_cell(markdown[3]), THEORY_TAG),
    ]


def install_convention_cells(path: Path, *, chinese: bool = False) -> None:
    """Replace any previous convention section without touching other cells."""

    notebook = nbf.read(path, as_version=4)
    notebook.cells = [
        cell
        for cell in notebook.cells
        if not ({THEORY_TAG, EXECUTABLE_TAG} & set(cell.metadata.get("tags", [])))
    ]
    notebook.cells[1:1] = convention_cells(chinese=chinese)
    nbf.write(notebook, path)


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    notebook_root = repository / "notebooks"
    paths = [notebook_root / "tutorials" / "RMP_resonance_analysis.ipynb"]
    paths.extend(
        notebook_root / "i18n" / language / "tutorials" / "RMP_resonance_analysis.ipynb"
        for language in LANGUAGES
    )
    for path in paths:
        install_convention_cells(path, chinese="/zh/" in path.as_posix())
        print(path.relative_to(repository))


if __name__ == "__main__":
    main()
