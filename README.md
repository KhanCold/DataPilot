# DataPilot

### 概要

一个简单的基于大模型的 CSV 数据分析系统。用户可上传 csv 格式的文件，然后输入自然语言指令如 “分析 Clothing 随时间变化的总销售额趋势” 。系统会自动生成数据分析 python 脚本，最终输出结构化的分析报告，包括大模型解释、运行结果、完整代码。

一个`规划器`用于将用户自然语言拆分成可执行计划，一个`执行器`用于生成对应步骤代码，一个`协调器`用于协调两个模块、评估结果是否满足用户最终需求以及生成分析报告。
更多实现细节请参考`代码简要说明.md`

### 安装所有必需的 Python 包:

```bash
cd backend
pip install -r requirements.txt
```

### 配置密钥

在 `backend` 目录下, 手动创建一个名为 `.env` 的文件。 添加你的 DashScope API 密钥（Qwen）, 如下所示:

```
DASHSCOPE_API_KEY="在这里填入你自己的API密钥"
```

### 运行程序

```bash
cd backend
python main.py
```

系统默认会读入测试 csv，也可以通过 input 来读入，具体修改在`main.py`第 10 行

```bash
# csv_paths_input = input("请输入要分析的 CSV 文件路径 (多个文件请用逗号分隔):\n> ")
csv_paths_input = "大模型实习项目测试.csv"
```

系统加载完毕会输出`[user]: ` 此时即可输入自然语言指令

测试样例：

1. 分析 Clothing 随时间变化的总销售额趋势.
2. 对 bikes 进行同样的分析.
3. 哪些年份 components 比 accessories 的总销售额高?

完成后, 会在控制台输出，并在 `result/` 目录下生成 Markdown 格式的分析报告。

### 样例输出

# DataPilot Analysis Report - Task 1

## Final Answer

1. **Analysis Overview**：
   本分析聚焦于服装（Clothing）类别的销售数据，首先从原始数据集中筛选出 Category 为 'Clothing' 的记录；随后对 Sales 列进行数据清洗，移除其中的货币符号（$）和千位分隔符（,），并将其转换为数值类型；最后按 Year 字段对清洗后的销售额进行分组汇总，计算每年的总销售额。

2. **Key Findings**：

- 2015 年服装类别的总销售额为 44,800.0；
- 2016 年总销售额增长至 60,400.0，同比增长约 34.8%；
- 2017 年总销售额显著上升至 104,200.0，较 2016 年增长约 72.5%，三年间呈现持续且加速增长的趋势。

3. **Conclusion**：
   服装类别的总销售额在 2015 年至 2017 年间呈现明显的上升趋势，且增长速度逐年加快，表明该品类在此期间市场需求持续扩大或销售策略成效显著。

## Execution Result

- Step 1 (completed): 筛选出 Category 为 'Clothing' 的数据，将 Sales 列中的货币符号和逗号去除并转换为数值类型，然后按 Year 分组计算总销售额。
  - Result:

```
Year
2015     44800.0
2016     60400.0
2017    104200.0
Name: Sales, dtype: float64

```

## Full Script

```python
import pandas as pd
df_1 = pd.read_csv('大模型实习项目测试.csv')
df_clothing = df_1[df_1['Category'] == 'Clothing'].copy()
df_clothing['Sales'] = df_clothing['Sales'].str.replace(r'[\$,]', '', regex=True).astype(float)
sales_by_year = df_clothing.groupby('Year')['Sales'].sum()
print(sales_by_year)
```
