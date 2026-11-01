# Transformer参考实现、训练入口与运行记录

本目录代码对应第21—43章。需要Python、PyTorch和Tokenizers。本次实际环境为PyTorch2.8.0、Tokenizers0.22.2，数值实验在CPU单线程运行。没有下载模型，没有执行195M正式训练、CUDA训练或DDP。

在本目录执行以下便宜核验：

```sh
python3 verify_transformer.py
python3 micro_experiments.py
python3 training_project.py demo --out demo_run
python3 train_195m.py --dry-run
python3 train_195m.py --dry-run --context 2048
```

`demo`会在指定目录生成自写教学JSONL、自训练320项字节级BPE、准备数据、运行30步tiny训练，并比较第15步保存恢复与连续训练。它生成临时检查点；本书只交付验证JSON，读者可以运行命令重新生成完整实验产物。

## 模型与数据接口

`ModelConfig()`精确定义195072768参数主模型，最大长度1024；改为2048不增加参数。`tiny_config()`使用V32、d32、2层、4头、SwiGLU64、长度32，共21664参数。BPE项目复用同一配置类，将V改成320，模型为30880参数。三者使用同种RoPE、预RMSNorm、无偏置SwiGLU、完整多头及输入输出权重绑定。

```python
from book_transformer import BookTransformer, tiny_config
model = BookTransformer(tiny_config())
out = model(input_ids, labels=labels, loss_mask=loss_mask,
            document_ids=document_ids)
```

input_ids与labels均为[B,T]。labels由调用方预先与各位置的下一token对齐，例如原序列[1,2,3,4]对应输入[1,2,3]与labels[2,3,4]。模型内部不再移位。labels中的-100不计分，loss_mask为可选0/1张量。document_ids用于可选同文档因果读取，不传时执行连续因果读取。返回logits、loss、loss_sum、token_count；没有labels时后三项为None。

全无效标签返回token_count=0与有计算图的零loss。训练器必须跳过optimizer.step与scheduler推进，不能仅对零loss照常更新；已有Adam状态或AdamW衰减仍可能使参数变化。配套CPU实验已在先执行有效更新、产生优化器状态后检查空更新：参数与优化器step均不变。

attention_backend默认reference，显式产生T×T分数与权重，分数和softmax局部关闭autocast并使用FP32。设为sdpa则调用PyTorch scaled_dot_product_attention；它不保证使用某个CUDA融合内核。外层仍有T×T布尔掩码。两种后端参数相同，小配置已对照，正式GPU内存与吞吐未测。

参数、embedding与残差主流默认FP32。CPU BF16 autocast实测部分线性输出和logits为BF16，RMSNorm及残差边界为FP32；不能将全部激活统一按2字节预算。

## 准备自己的数据与显式单GPU入口

JSONL每条记录必须含id、text、source、license。脚本检查精确正文与id跨集合交集、保留标记字面串、词表实际大小与产物哈希；没有实现近重复检测或自动核实使用依据。正式准备示例：

```sh
python3 training_project.py prepare --train-jsonl data/train.jsonl --valid-jsonl data/valid.jsonl --out prepared --vocab-size 32768 --length 1024
```

训练语料太少、无法产出32768项时会报错，不静默缩小。prepare将dataset.pt完整保存为张量，训练入口把它加载到CPU内存；这不是多Btoken流式读取器。CPU `training_project.py train`只允许tiny且窗口不超过64；可用demo生成的prepared目录继续实验。

`train_195m.py`是独立单CUDA设备入口。真正启动必须显式指定数据、输出、设备和更新预算；它提供FP32或设备支持时BF16、SDPA、可选块检查点、有效token累计、预热余弦调度、周期保存和恢复。下面是未执行的正式路线示例：

```sh
python3 train_195m.py --data prepared --out runs/195m-01 --device cuda:0 --max-updates 1000 --max-train-tokens 50000000 --micro-batch 1 --accumulation 8 --precision bf16 --warmup-updates 100 --checkpoint-blocks --save-every 100
```

两个预算上限谁先到就停止，下一整步会超过token上限时停止并还原预取游标，不超额执行。上述1000步满载约8.19M目标，50M是额外上限。当前只实际执行dry-run与CPU链路，CUDA容量、内核、信号停止路径与正式长训练均未实测。此入口没有实现DDP，不能直接多进程重复启动共享输出。

## 保存恢复的调用契约

`save_checkpoint`与`load_checkpoint`在模型、优化器、配置、成功step、训练token、采样器状态与Python/Torch随机状态之外，支持可选的scheduler、scaler、metadata。调用方必须在保存与恢复时传入对应对象；传入expected_metadata才会比较数据与计划身份。两侧遗漏传参不会自动恢复外部状态。正式入口传入scheduler与manifest/计划绑定，BF16路径无scaler；未来FP16训练器需自行接入scaler。NumPy随机流、数据工作进程与分布式rank游标没有自动保存。

仅在完成更新并清零grad的边界保存。中途累计恢复所需梯度与分母没有实现。只加载可信检查点；当前完整状态读取不是陌生文件的安全导入器。

文件先写同目录临时路径、flush、文件fsync，再os.replace。它提供单写者基础原子替换，没有父目录fsync、自动多版本保留或跨文件系统断电持久性保证。异步快照、分片提交与DDP保存仍需扩展并演练。

## 本次交付的验证记录

- `transformer_verification.json`：195M仅meta核数；tiny21664参数真实80步，损失3.438835降至0.000810；因果/文档隔离、恢复下一更新与块检查点梯度检查通过；SDPA对照误差见JSON。
- `micro_experiments.json`：21152参数、六位置远处复制100步，损失2.874831降至0.000844；关键首token替换后预测对应改变。另含实际CPU BF16模块类型记录，不是GPU验证。
- `project_verification.json`：自训练BPE320、训练窗口213/验证37，数据有效目标6700/1164；tiny30880参数30步、3789个有效训练目标，训练损失5.773962降至2.950416；连续与恢复参数差0。验证3.003730只来自相似教学模板，不代表自然文本泛化。
- `train_195m_dry_run_1024.json`与`train_195m_dry_run_2048.json`：两个上下文配置都只在meta核得195072768参数。

官方接口依据与教学选择的核对说明位于书章和本项目来源记录；版本升级或改变后端后，应重新执行上述核验，再测目标设备。
