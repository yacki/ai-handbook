# 配套代码与实验入口

这是教材的参考代码和离线实验，不包含预训练权重。约195M模型配置由本书设计；模型只做meta参数核对，实际训练记录来自tiny模型或小型人工任务。

建议使用独立Python环境。已运行的张量实验使用PyTorch 2.8.0与CPU，BPE实验使用Tokenizers 0.22.2；不需要为阅读教材下载大型模型。各文件开头说明依赖和用途，运行前先查看该文件的入口与默认配置。

| 文件 | 用途 | 运行范围 |
|---|---|---|
| ch13_autograd.py | 手算与自动求导对照 | CPU小型线性回归 |
| ch19_bigram.py | 25参数bigram训练与生成 | CPU、人工数据 |
| book_transformer.py | 全书统一Transformer与tiny配置 | 默认195M配置，勿把默认配置当成已训练权重 |
| verify_transformer.py | 注意力、因果/文档边界、tiny训练、恢复 | CPU tiny与meta参数核对 |
| training_project.py | 自训练BPE、来源清单、文档打包、累计梯度与中断恢复 | 默认教学流程为CPU微型模型；正式词表准备需自备足够语料 |
| train_195m.py | 正式单CUDA设备训练入口，数据与配置身份、预算、恢复与调度 | 本次只运行dry-run与meta核数；CUDA训练未经本书实测，不包含DDP |
| micro_experiments.py | 延迟复制任务、CPU BF16类型追踪 | 人工小任务；不测自然语言泛化 |
| ch44_65_lab.py | DPO、奖励、蒸馏、内存、量化、分页、推测采样等13组算例 | 标准库算术与采样；不是完整训练或GPU内核 |
| ch47_48_tiny_adaptation.py | SFT与LoRA、冻结权重与合并核对 | CPU tiny模型与人工序列 |
| ch52_56_budget_design.py、ch56_ablation_lab.py | 配比、人工缩放曲线、消融统计 | 教学数字与确定性模拟 |
| ch58_sampling_lab.py | 温度、top-k、top-p与频率核对 | 标准库随机采样 |
| ch59_62_cache_lab.py | prefill/decode、RoPE偏移、KV缓存与GQA映射 | CPU数值等价；不测GPU性能 |
| ch62_online_softmax_lab.py | 分块softmax归并 | 标准库数值检查；不运行FlashAttention内核 |
| ch63_quantization_lab.py | 分组量化误差、异常值与tiny输出对照 | CPU浮点模拟；不是原生低位内核 |
| ch64_scheduler_lab.py、ch64_copy_on_write_lab.py | 连续准入、页面、共享前缀与写时复制 | 标准库确定性模拟 |
| ch65_speculative_edge_lab.py | 拒绝采样与零概率边界 | 标准库算术；不测GPU加速 |
| structured_tools_demo.py | 结构化工具参数与范围校验 | 标准库、只读虚构设备 |
| offline_rag.py | 检索、过滤、证据摘录与无答案 | 标准库、虚构手册 |
| agent_state_machine.py | 预算、取消、失败与证据不足终态 | 标准库、确定性控制器 |
| streaming_lab.py | SSE拆包、ASGI断开、队列背压、取消与回收 | 标准库、模拟引擎与虚拟传输 |
| harness_recovery_lab.py | 事件去重、订阅直通、工具回执恢复 | 标准库、临时SQLite、虚构业务 |
| verify_system_examples.py | 应用章节算术与三个离线程序 | 标准库、确定性检查 |
| security_boundary_demo.py | 证据内容、读取范围与工具权限边界 | 标准库、虚构请求 |
| multimodal_arithmetic_demo.py、architecture_arithmetic_demo.py | 图像片段、路由容量、状态递推与扫描算术 | 标准库；不运行真实多模态、MoE或Mamba模型 |
| reasoning_budget_demo.py | 奖励标准化、候选覆盖与验证器限制 | 标准库；不运行R1 |
| model_metadata_demo.py | 本地Qwen3-8B配置快照的形状与缓存账 | 标准库；不下载或加载模型权重 |
| maintenance_product_demo.py | 集成离线维护示例与五类终态验收 | 标准库、固定虚构资料；不调用语言模型或真实设备 |
| verify_frontier_examples.py | 前沿章节的21项算术及六个程序核验 | 标准库；不加载公开大模型 |

各章提供对应入口。版本清单见requirements-teaching.txt，模型与数据项目的细节见[Transformer运行说明](Transformer运行说明.md)。请将教学实现、实际运行记录与原论文全部实验分开理解。

装好对应设备的PyTorch与Tokenizers后，可以在本目录先走微型模型路线：

```sh
python3 verify_transformer.py
python3 training_project.py demo --out ./runs/bpe-demo
python3 train_195m.py --dry-run
```

第一条运行tiny模型检查，第二条产生人工语料、分词器和微型训练/恢复记录，第三条只核对195M配置。正式训练需要自行准备已获许可的语料和CUDA设备，显式传入数据、产物位置与更新预算；该入口一次把准备好的数据载入CPU内存，不能当作多B token流式数据系统。

普通标准库实验可直接运行，例如：

```sh
python3 streaming_lab.py
python3 harness_recovery_lab.py
```

这些命令不启动公开服务器，不连接外部账号，不发送真实消息。若要接入真实模型引擎或业务工具，需要依照对应章节补足版本适配、权限、容量与故障验证。
