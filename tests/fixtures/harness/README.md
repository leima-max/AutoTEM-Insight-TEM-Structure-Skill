# Harness 解析验收样例

首批 HRTEM、HAADF、直接衍射和应变样例由发行包中的 [cases.py](../../../autotem/acceptance/cases.py) 在新的验收目录生成。每次运行记录生成公式、轴、数据类型、标定来源和输入哈希。

附加的原子操作、标尺及可选模拟样例使用维护者脚本旁的[样例清单](../../../tools/acceptance/operation_variants.json)。源码测试与发行环境读取同一份清单并调用相同的数值检查。
