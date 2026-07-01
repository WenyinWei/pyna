工作流与扩展辅助层
==================

``pyna.topo`` 的辅助层解决“怎么把外部模型接入 pyna”的问题：

- ``TopologyWorkflow``：notebook 和脚本优先使用的门面；
- ``protocols``：定义 FlowLike、MapLike、SectionLike 等结构契约；
- ``adapters``：把数组、求解器输出或外部对象规范化；
- ``builders``：集中处理闭合检查、metadata 和对象提升；
- ``bridges``：把 ``Cycle/Tube`` 切到 ``PeriodicOrbit/IslandChain``；
- ``factories``：为配置驱动项目提供稳定字符串 key。

使用原则：adapter 只做格式归一；builder 才表达“这是闭合对象”之类的数学
声明；factory 只在需要稳定 key 或后端选择时引入。

英文详细 API：

- :doc:`../../en/api/dynamics-patterns`
