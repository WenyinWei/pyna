扩展模式
========

辅助类不是为了设计模式本身存在，而是为了让 notebook 和下游库少写脆弱胶水。

推荐使用顺序
------------

1. 普通脚本和教学案例先用 ``TopologyWorkflow``；
2. 外部对象只差少量字段时，用 protocol 约束接口；
3. 数据格式不统一时，用 adapter 归一到核心对象；
4. 需要闭合检查、metadata 和回链时，用 builder；
5. 连续对象切到截面离散对象时，用 bridge；
6. 配置驱动或后端可切换时，再引入 factory/registry。

反模式
------

- adapter 静默把开放轨道变成 ``Cycle``；
- factory 只包了一层普通构造函数却没有稳定 key 或后端选择价值；
- notebook 里到处直接拼装 dataclass，导致闭合容差和 metadata 分散。

英文完整版本：

- :doc:`../../en/development/dynamics-patterns`
