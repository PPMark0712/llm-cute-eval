# lm-cute-eval：一个轻量级的大语言模型评测框架

## 模块介绍

### 任务总模块

包含若干单一任务模块，以及2个脚本文件

```
__init__.py: 包含LOAD_TASK_DATA和MATCH_TASK_ANSWER全局字典变量，LOAD_TASK_DATA是{任务名: 对应加载数据的函数}的映射，MATCH_TASK_ANSWER是{任务名: 对应匹配答案的函数}的映射
match_answer.py: 包含一些常用匹配答案函数
若干名称为{task_name}的文件夹，表示每个任务有一个独立的模块
```

### 单一任务模块

每个评测任务都有一个独立的文件夹，文件夹名称为任务名称（例如mmlu、gsm8k、commonsenseqa），在文件夹中有3个文件：

```
__init__.py: 空的文件
load_data_{task_name}.py: 读取数据集
包含load_data_{task_name}(args)函数，返回任务数据
match_answer_{task_name}.py: 匹配模型输出答案，返回结果
包含def match_answer_{task_name}(infer_result:dict, round_idx:int, args)函数
```

### model format模块

每种模型类型有一个脚本，用于控制prompt格式和多轮对话格式，默认为default_format（没有格式，保留原始输入）

```
__init__.py: 包含MODEL_FORMAT全局变量字典，该字典是{模型类型: 模型格式化函数}的映射
{model_type}_format.py: {model_type}类型的模型格式，包含format_prompt_{model_type}格式
```

## 主要数据结构

在本框架中，有两个主要数据结构，分别是**任务数据结构**以及**结果数据结构**

### 任务数据结构

读取数据以及推理结果都共享一个数据结构，其结构如下：

```
类型: Dict[str, Dict[str, List[dict]]]
{任务名: {子任务名: [第一条数据, 第二条数据]}}

以下为一个具体的实例
{
	"commonsenseqa": {
		"commonsenseqa": [  ## 若没有子任务，则字典的第二层只包含一个和任务名称相同的子任务名
			第一条数据(dict), 第二条数据(dict), ...
		]
	}
	“mmlu”: {
		“abstract_algebra": [
			第一条数据(dict), 第二条数据(dict), ...
		],
		"anatomy_test": [
			第一条数据(dict), 第二条数据(dict), ...
		],
		...
	}
}
```

单条数据的结构如下，在第一轮推理中，模型实际输入为instruction + fewshot_prompt + prompt_round1：

```
{
	**item: 从数据集中读取的内容，该部分可以灵活调整，主要用于匹配答案,
	"instruction": str,
	"fewshot_prompt": str,
    "prompt_round1": str,
}
```

### 推理结果数据结构

经过推理后，任务数据的单条数据中会增加一个infer_round1字段，表示模型输出：

```
{
	**item: 从数据集中读取的内容，该部分可以灵活调整，主要用于匹配答案,
	"instruction": str,
	"fewshot_prompt": str,
    "prompt_round1": str,
    "infer_round1": str,
}
```

### 分数数据结构

```
类型: Dict[int, Dict[str, dict]]
{round_idx: {task_name: task_result}}

以下为一个具体的实例
{
	"round1": {
        "arc": {
            "arc_e": 0.5,
            "arc_c": 0.3
        },
        "commonsenseqa": {
            "acc": 0.5
        },
        "gsm8k": {
            "exact_match": 0.1,
            "flexible_match": 0.2
        },
    }
}
只有一轮时，输出结果summary.json中会直接显示"round1"对应的dict
```



## 运行逻辑

1、初始化

获取运行参数，确定评测任务和输出目录

2、读取数据

该框架先读取数据，再加载模型，避免加载耗时较长时难以调试数据。读取数据时，调用所有需要评测的任务对应的读取函数

3、加载模型

加载模型并初始化vllm库中的LLM以及sampling\_prams对象，若运行参数不指定sampling\_prams，则使用默认值（于model.py中）

4、推理（支持多轮）

将所有数据加载完毕后进行批量推理，保存推理结果

5、匹配答案

将分别调用评测任务的匹配答案函数，传入推理结果，返回对应评测指标

6、保存结果

将结果整合后，保存到对应文件夹，其中包含每个数据集的每个子任务的细节，以及summary，然后退出程序。



## 数据集扩展

当需要添加一个数据集时，需要根据以下步骤修改程序，假设新增的任务名称为"newtask"

### 1、存放数据集

在data/tasks文件夹中新建newtask文件夹，并放入数据

### 2、新建单一任务模块

```
在lm_cute_eval/tasks文件夹中新建newtask文件夹，并创建四个文件：
__init__.py: 空文件
load_data_newtask.py: 编写加载数据代码，包含load_data_{task_name}(args)函数
match_answer_newtask.py: 编写匹配答案代码，包含def match_answer_{task_name}(infer_result:dict, round_idx:int, args)函数
config_newtask.json: 编写当前任务的配置
```

以上新建的脚本文件可以仿照其他任务，配置文件可以随意添加想要的配置，任务之间互不影响。此外，也可以在该目录中新建其他代码。

### 3、更新任务模块总配置

```
在lm_cute_eval/tasks/__init__.py中新增import内容，并更新LOAD_TASK_DATA和MATCH_TASK_ANSWER字典。可以参考文件中的代码，新增一个任务时，import文件的顺序以级字典中key的顺序应当按字典序升序排序，以便于维护。
```

### 4、更新任务总模块

在./lm_cute_eval/utils.py中将newtask加入TASK_LIST。











