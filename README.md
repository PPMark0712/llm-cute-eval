# lm-cute-eval：一个轻量级的大语言模型评测框架

这是一个轻量级的大语言模型评测框架，目前支持少量常用评测集，其优点在于不同任务模块之间解耦，扩展性强，可以较方便地添加新的任务。该评测框架使用transformers和vllm库进行推理。

## 开始运行

配置环境（用较低版本的库也可以运行，但是没有经过测试）

```
pip install -r requirements.txt
```

下载数据，并将数据文件data.zip解压到项目根目录

```
cd llm-cute-eval-main
wget https://github.com/PPMark0712/llm-cute-eval/releases/download/0.2/data.zip
unzip data.zip
```

编辑run.sh脚本，需要考虑的参数如下：

```
模型配置
model_path: 模型绝对路径
model_type: 模型变量类型，默认vllm，目前可选vllm和hf
format_type: 模型类型，用于控制prompt格式，默认default

任务配置
tasks: 需要评测的任务名称，用空格隔开，例如需要评测mmlu和gsm8k，则在命令中加入--tasks gsm8k mmlu，也可以包含一个all，自动评测所有任务。
config_path: 任务配置文件路径，缺失值自动填充为对应任务文件夹中的默认config。
data_path(不需要修改): 数据集路径。

保存配置
output_path: 输出目录，默认output
save_name: 输出文件夹名称。
save_infer_results: 保存推理结果，而非只保存一个分数
save_infer_texts: 保存便于阅读的输入输出文本到infer_result{round_idx}.txt
no_timestamp: 输出文件夹不包含时间，若包含时间，则会保存到"./output/{time}_{model_name}/"中
temp_file_path(不需要修改): 临时文件保存目录，主要用于humaneval评测集。

推理配置
rounds: 推理轮数（用于其他实验，需要自己控制中间对话的prompt，具体见lm_cute_eval/get_multiround_prompt.py。
seed: 随机种子。
use_cpu(不需要使用): 使用CPU推理(用于debug)。
temperature:模型推理参数
top_p:模型推理参数
top_k:模型推理参数
max_new_tokens: 最多生成的token数量，默认160，不同数据集不一样，且本框架不可以分开设置每个任务的new token数量，所以取了个较大的值。
```



例如你想用mmlu和gsm8k评测两个模型：

```bash
export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false

declare -A models=(
   	["model_name1"]="model_path1"
	["model_name2"]="model_path2"
)

for model_name in "${!models[@]}"; do
    model_path=${models[$model_name]}
    python main.py \
        --model_path "$model_path" \
        --model_type vllm \
        --format_type default \
        --tasks mmlu gsm8k \
        --save_name "$model_name" \
        --save_infer_texts \
        --save_infer_results \
        --config_path "config.json" \
        --output_path output/debug \
        --max_new_tokens 180 \
        --temperature 0.1 \
        --top_p 0.2 \
        --top_k 20 \

done

```

配置config：

根目录下，有默认config.json文件，在每个数据集的config文件中也又默认值，可以根据需要来修改config中的内容，其格式如下：

```
{
    task_name_1: task_config_1,
    task_name_2: taks_config_2,
    ...
}

例如:
{
    "gsm8k": {
        "num_fewshot": 8,
        "limit": 0
    }
    "mmlu": {
        "num_fewshot": 5,
        "limit": null
    }
}
```

若某一项的config为空，则会自动使用对应任务的模块中的config文件

```
limit: (int) 只评测改数据集的前几条。若为0或null，则全量评测；若改数据集有子任务，则表示每个子任务读取limit条数据。
num_fewshot: (int) fewshot数量，可以使用默认值，部分数据集的fewshot有取值范围。
subjects: (list) 需要评测的子任务的名称列表，例如mmlu中有abstract_algebra, anatomy_test
```



## 评测任务介绍

### arc

数据集来源：使用[opencompass](https://github.com/open-compass/opencompass)中的数据。

评测指标：匹配回答中的第一个选项，判断是否正确。

### commonsenseqa

数据集来源：使用[opencompass](https://github.com/open-compass/opencompass)中的数据。

评测指标：匹配回答中的第一个选项，判断是否正确。

### gsm8k

数据集来源：使用huggingface中的数据，fewshot prompt采用了[cot-hub](https://github.com/FranxYao/chain-of-thought-hub)中的部分数据。

评测指标：

exact_match：匹配回答中和四个'#'后面的数字，该数字正确则为正确。

flexible_match：匹配回答中任何数字，有一个正确则为正确。

### hellaswag

数据集来源：使用huggingface中的数据。

评测指标：匹配回答中的第一个选项，判断是否正确。

### humaneval

数据集来源：使用huggingface中的数据，不能设置fewshot，但是实际使用了2-shot来控制输出格式，便于提取代码块，这个2-shot prompt。

评测指标：找到第一个代码块，并用humaneval库中的评测代码进行评测

### icleval

数据集来源：[ICLEval/data/tasks_data](https://github.com/yiye3/ICLEval/tree/main/data/tasks_data)，修复了部分文件中"examples"写成"exmaples"的拼写错误。

评测指标：若标准答案是模型生成文本的子串，则为正确。

### mmlu

数据集来源：使用[opencompass](https://github.com/open-compass/opencompass)中的数据，以级[cot-hub](https://github.com/FranxYao/chain-of-thought-hub)中的fewshot_cot_prompt。

评测指标：匹配回答中的第一个选项，判断是否正确。

### winogrande

数据集来源：使用[opencompass](https://github.com/open-compass/opencompass)中的数据，用GPT4生成的fewshot_cot_prompt。

评测指标：匹配回答中的第一个选项，判断是否正确。

### rgb

数据集来源：[chen700564/RGB](https://github.com/chen700564)

评测指标：有一个可能的回答出现在答案里即为正确。

### xsum

数据集来源：[新建标签页 (github.com)](https://github.com/EdinburghNLP/XSum)

评测指标：用BAAI/bge-m3模型计算和答案的相似度。

注意，该功能由于依赖包较复杂，暂未完善，全部被注释掉了，自己配好环境是可以用的。