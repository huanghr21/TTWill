# 中文零售种子与参考任务

`source/` 保存 τ²-bench 固定提交
`c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4` 的原始文件：

| 本地文件 | 上游路径 | 用途 |
| --- | --- | --- |
| `tau2_zh_retail.json` | `data/tau2/multilingual/zh/regional/retail.json` | 500 用户、1000 订单、50 类商品及 591 个规格；官方中文 50 题、政策与本地化元数据 |
| `retail_50.json` | `data/tau2/task_subsets/retail_50.json` | 冻结任务 ID 清单 |
| `tau2_zh_pack.yaml` | `data/tau2/multilingual/zh/pack.yaml` | 官方中文人设、语言要求及文字对话开场白 |
| `tau2_retail_tasks.json` | `data/tau2/domains/retail/tasks.json` | 全部 114 道原始任务，排除 retail_50 后得到 64 道开发题 |
| `tau2_retail_database.json` | `data/tau2/domains/retail/db.json` | 官方原始业务数据，供转换和映射校验 |
| `tau2_retail_policy.md` | `data/tau2/domains/retail/policy.md` | 官方原始零售政策，供固定版本编译校验 |
| `tau2_zh_retail_localization.json` | `data/tau2/multilingual/zh/localization/retail.json` | 官方中文映射表 |
| `LICENSE.tau2` | `LICENSE` | 上游 MIT 许可证 |
| `manifest.json` | 本项目生成 | 来源提交、转换版本、各文件 SHA-256 及字节数 |

八份上游文件原样复制；中文种子导入由 `ttwill.bench.retail.load_retail_bundle` 校验原有四份文件的内容哈希和任务集合，开发转换另校验新增四份来源。
数据中的人物、地址和联系方式是基准合成数据。中文包金额使用上游固定的人民币换算，
没有实时汇率请求。

语言包用于第 7 步官方用户模拟器。文字对话只读取包内的中文人设、零售文字说明和开场白，
人设选择复用官方按任务元信息与种子分配的规则；零售任务元信息已完整包含在
`tau2_zh_retail.json` 中，与上游 `retail_metadata_zh.json` 一致。语言包内其他领域和语音
资源引用不在本步加载；第 9 步只补充零售转换所需的原始任务、数据库、政策和映射表。所有复制内容适用
已保存的上游 MIT 许可证，语言包与原有数据库来自同一提交。

重新取得同一份文件时，克隆
[上游仓库](https://github.com/sierra-research/tau2-bench)，检出上述提交后复制表中路径；
不要重新编译或手动修改中文包。只有明确更新基准版本时才更新来源清单和校验值。

业务库初始化只读取中文包的 `database`，不需要安装 τ²。重放校验需要单独的 `bench`
依赖：正常导入上游 `RetailDB`、`RetailTools`，按题新建独立 SQLite 库，两边执行相同
参考业务动作，然后调用官方 `RetailDB.get_hash()` 比较完整终态。官方参考轨迹省略身份
查询的任务使用目标记录所属用户作为测试夹具身份；这不代表 Agent 的认证流程已经通过。

重放还比较七个业务读工具和参考动作返回值；通用计算器与转人工只记录官方结果，因为
它们没有业务库副作用。只读错误查询在两侧均被拒绝时记为一致，保留各自错误信息。
所有终态差异都保留在报告中，不替换官方结果、不修改哈希、不排除特定题目。

生成的 SQLite 库、完整报告与运行轨迹放在 `runs/`，不作为来源文件提交。

第 6 步的[本体场景验收](../tests/acceptance/README.md)从已校验的数据库中抽取合成用户
`junjie_liu_6635` 及其四张订单、所需商品作为小型开发种子，使用同源政策。
用户原话和模型工具提议由项目手写，不使用官方题目、参考动作或判分答案；每项独立导入业务库，
不改这里的源数据。这些开发脚本不作为冻结困难评测集或真实模型效果报告。

## 开发任务与固定快速集

`tasks/` 保存可复现的转换结果与冻结清单，开发题来自上述固定提交的 114 题减去 `retail_50` 的 64 题，使用同一中文业务种子：

| 文件 | 内容 |
| --- | --- |
| `localization_overrides.json` | 官方映射未覆盖字段的人工审阅补充，作为转换输入记录内容摘要 |
| `conversion_report.json` | 官方中文 50 题校验、开发题转换及待审阅字段的报告 |
| `dev_64.json` | 开发题、任务元信息、转换版本和逐项来源摘要 |
| `dev_quick.json` | 固定 20 题及顺序、抽样种子、分层配额、覆盖统计和开发集摘要 |

转换和抽样均不调用模型。首次生成与参考重放的命令为：

```bash
uv run --extra bench ttwill convert-tasks --source data/source --output data/tasks
uv run --extra bench ttwill freeze-quick-set --tasks data/tasks --seed 0 --size 20
uv run --extra bench ttwill bench --dataset dev_64 --output runs/dev-reference-001
```

已有转换产物或快速清单会拒绝覆盖。复现到新目录时，先复制人工审阅补充文件，再将新目录传给转换的 `--output`、抽样与重放的 `--tasks`。转换必须通过官方中文 50 题校验并清空待审阅字段；人工补充只作用于明确记录的字段，不修改官方来源文件。

快速集按“写操作类型 × 次数”分层，默认种子 `0`、数量 `20`，允许 15～20 题。清单生成后固定，日常修改按相同题目配对比较，较大改动运行全部 64 题。64 题实际不含支付方式修改，所以快速集覆盖其余 6 种写工具以及只读、多写任务；支付修改另有无模型测试。快速集仍属于开发集，不代替 `retail_50` 或困难评测集。

模型批次使用 `ttwill evaluate-batch --dataset dev_quick --user-tier flash --judge-tier flash --output runs/dev-quick-001`，完整参数与报告口径见 [项目 README](../README.md#开发任务与批量评测)。每题一次、各用独立状态，不自动补跑；来源摘要和题单随批次写入 `plan.json`，完成率与已完成样本成功率分开报告。
