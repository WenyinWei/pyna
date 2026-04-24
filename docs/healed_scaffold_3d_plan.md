# Healed Scaffold 3D — 最小重构路线图

## 目标

把当前脚本里的

- 参考截面构造
- 沿真实磁场线传播到其它 toroidal sections
- 由传播后的点重建各截面 surface family

从一次性脚本逻辑，提升为 `pyna.topo` 里的可复用基础设施。

---

## 现状判断

当前 `IslandHealedCoordMap` 的主要问题不是局部拟合精度，而是：

1. 每个 section 基本独立定义自己的 poloidal parameterization；
2. `eval_RZ()` 仍依赖 nearest-section 的局部 Fourier 外推；
3. `r=1` 边界仍是 section-wise 对象，而非统一 3D 对象。

因此真正需要先解决的是 **3D correspondence**，而不是继续给每个截面单独补丁。

---

## Phase 1（本次已开始）

### 新增模块

`pyna/topo/healed_scaffold_3d.py`

### 已提供对象

- `TransportedSection`
  - 一个离散 toroidal section 上的 transported `(R,Z)` 点阵 + valid mask
- `SectionCorrespondence`
  - 显式表示 reference section 与目标 section 的 transport correspondence
- `FieldLineScaffold3D`
  - 由 reference section 出发构造的离散 3D scaffold
- `trace_grid_to_phi`
  - transport 一个二维 `(r,θ)` 采样点阵
- `trace_section_curve_to_phi`
  - transport 一条截面曲线（如 r=1 boundary）
- `trace_surface_family_to_sections`
  - 函数式封装，直接返回 `FieldLineScaffold3D`

### 设计定位

这是一个**离散 3D scaffold 层**，不是最终的连续坐标系终态。

先解决：
- 脚本复用
- 3D sample correspondence
- 统一的 traced data container

暂不解决：
- 连续 φ 插值的高阶光滑性
- 全局 inverse map
- variational / ghost-surface reconstruction

---

## 推荐下一步（Phase 2）

### 把 W7-X 脚本改为调用新模块

优先改：

`topoquest/scripts/w7x/w7x_healed_scaffold.py`

重构方式：
1. 保留参考截面拟合逻辑；
2. 把 `_trace_pts(...)` 替换为传给 `FieldLineScaffold3D.from_reference_map(...)` 的 `trace_func`；
3. 用 `FieldLineScaffold3D.section_at(phi)` 提取各 section 的 transported grids；
4. 脚本内只保留“如何从 transported point cloud 重新拟合本 section 表示”的部分。

这样做的收益：
- 先不动科学流程；
- 先把 transport correspondence 正式对象化；
- 脚本里最容易出错的 glue code 被去重。

---

## 推荐后续（Phase 3）

### 新增 `IslandHealedCoordMap3D`

建议定位：
- forward map 优先
- inverse 先做近似版

核心变化：
- 不再用 `_nearest_section(phi)` 作为几何底层
- 改用 `FieldLineScaffold3D` 作为底层 embedding
- section-wise Fourier 退化为：
  - 参考截面构造器
  - fallback / bootstrap 工具

---

## 推荐后续（Phase 4）

### 边界对象独立化

新增建议对象：
- `HealedBoundary3D`

职责：
- 从参考截面的 healed `r=1` / `C_XO` 出发
- 用 field-line transport 得到统一 3D boundary
- 提供：
  - `sample_section(phi)`
  - `distance_to_boundary(...)`
  - `fit_section_fourier(...)`

---

## 推荐后续（Phase 5）

### inverse map 重写

目前 `Nelder-Mead` 在岛链 / separatrix 附近容易掉 branch。

建议路线：
1. 离散 scaffold 网格上做 coarse nearest guess；
2. 用局部 least-squares / Newton refinement；
3. 用 transport continuity 做 warm start。

这是在 forward 3D scaffold 稳定之后再做的事。

---

## 为什么这个顺序是对的

因为当前瓶颈不是：
- 截面内 Fourier 不够高阶
- 或 X/O 约束不够强

而是：
- **同一个 `(r,θ)` 在不同 φ 上没有被统一视为同一个三维几何对象**。

所以必须先让 3D correspondence 成立，再谈更高级的 healing。
