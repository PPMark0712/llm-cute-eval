# lm-cute-eval：一个轻量级的大语言模型评测框架

这是一个轻量级的大语言模型评测框架，目前支持少量常用评测集，其优点在于不同任务模块之间解耦，扩展性强，可以较方便地添加新的任务。该评测框架使用vllm库进行推理，暂时不支持使用transformers自带的推理函数。

## 开始运行

配置环境（用较低版本的库也可以运行，但是没有经过测试）

```
pip install -r requirements.txt
```

下载数据，并将数据文件data.zip解压到项目根目录

```
cd lm-cute-eval-main
wget https://github.com/PPMark0712/lm-cute-eval/releases/download/0.1/data.zip
unzip data.zip
```

编辑run.sh脚本，需要考虑的参数如下：

```
model_path: 模型绝对路径
model_type: 模型类型，用于控制prompt格式，默认default
sampling_params: vllm推理框架使用的参数
tasks: 需要评测的任务名称，用空格隔开，例如需要评测mmlu和gsm8k，则在命令中加入--tasks gsm8k mmlu
save_name: 输出文件夹名称
save_infer_results: 保存推理结果，而非只保存一个分数
config_path: 任务配置文件路径
output_path: 输出目录，默认./output
no_timestamp: 输出文件夹不包含时间，若包含时间，则会保存到"./output/{time}_{model_name}/"中
```

可能不需要考虑的参数：

```
rounds: 推理轮数，默认1，仅用于其他实验
refine_prompt: 多轮推理过程中的prompt，仅用于其他实验
temp_file_path: 临时文件保存目录，主要用于humaneval评测集
```

例如你想要评测mmlu和gsm8k：

```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

declare -A models=(
    ["model_name"]="model_path"
)

for model_name in "${!models[@]}"; do
    model_path=${models[$model_name]}
    python main.py \
        --model_path "$model_path" \
        --model_type default \
        --tasks gsm8k mmlu \
        --save_name "$model_name" \
        --save_infer_results \
        --config_path config.json \
        --output_path output
done
```

配置config：

根目录下，有默认config.json文件，可以根据需要来修改config中的内容，其格式如下：

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



## 评测任务细节

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

数据集来源：使用原始数据（[ICLEval/data/tasks_data](https://github.com/yiye3/ICLEval/tree/main/data/tasks_data)），修复了部分文件中的错误，且不支持copy_dict_search_string.json、copy_natural_language_string.json这两个子任务。

评测指标：若标准答案是模型生成文本的子串，则为正确。

### mmlu

数据集来源：使用[opencompass](https://github.com/open-compass/opencompass)中的数据，以级[cot-hub](https://github.com/FranxYao/chain-of-thought-hub)中的fewshot_cot_prompt。

评测指标：匹配回答中的第一个选项，判断是否正确。

### winogrande

数据集来源：使用[opencompass](https://github.com/open-compass/opencompass)中的数据，用GPT4生成的fewshot_cot_prompt。

评测指标：匹配回答中的第一个选项，判断是否正确。

