动力系统工作流和扩展 Helper
============================

``pyna.topo`` 在核心拓扑对象之上提供 workflow helper。主要面向用户的入口是
``TopologyWorkflow``；低层 Protocol、Adapter、Builder、Bridge 和 Factory 模块
仍然提供给需要稳定扩展点的下游库。

Workflow Facade
---------------

``TopologyWorkflow`` 面向 notebook 和日常脚本。它组合系统构造、积分/迭代、显式提升
和截面切割，而不引入新的数学对象类型。

.. automodule:: pyna.topo.workflow
   :no-index:
   :members:
   :show-inheritance:

Protocol
--------

结构化 protocol 描述外部系统的扩展契约。第三方对象只要实现所需属性和方法即可参与
工作；不强制继承 pyna 类。

.. automodule:: pyna.topo.protocols
   :no-index:
   :members:
   :show-inheritance:

Adapter
-------

adapter 将数组、solver 输出和已有 pyna 对象规范化为核心几何表示。它们不会静默地把
采样数据提升为不变对象。

.. automodule:: pyna.topo.adapters
   :no-index:
   :members:
   :show-inheritance:

Builder
-------

builder 编码显式提升规则。例如，只有通过可要求闭合样本的 builder 或 adapter 调用，
轨迹才能被提升为 ``Cycle``。

.. automodule:: pyna.topo.builders
   :no-index:
   :members:
   :show-inheritance:

Bridge
------

bridge 连接连续时间和离散时间对象族：``Cycle -> PeriodicOrbit`` 以及
``Tube/TubeChain -> IslandChain``。

.. automodule:: pyna.topo.bridges
   :no-index:
   :members:
   :show-inheritance:

Factory 和 Registry
--------------------

factory 为系统、几何对象和 Poincare 映射提供稳定构造入口。registry 是显式且防重复
的，因此测试和下游库可以隔离自己的扩展。

.. automodule:: pyna.topo.factories
   :no-index:
   :members:
   :show-inheritance:
