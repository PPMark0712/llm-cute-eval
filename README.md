# llm-cute-eval：一个轻量级的大语言模型评测框架

[![GitHub Repo stars](https://img.shields.io/github/stars/PPMark0712/llm-cute-eval?style=social)](https://github.com/PPMark0712/llm-cute-eval/stargazers)

这是一个轻量级的大语言模型评测框架，目前支持常用评测集，其优点在于不同任务模块之间解耦，扩展性强，可以较方便地添加新的任务，并且可以支持多轮推理。该评测框架使用vllm库进行加速推理。该框架仅支持生成式评测，不支持困惑度评测方式，因此不适合生成能力较差的小模型。


- [开始运行](#开始运行)
    - [配置环境](#配置环境)
    - [下载数据](#下载数据)
    - [运行参数](#运行参数)
    - [运行脚本](#运行脚本)
    - [多进程多卡运行脚本（可选）](#多进程运行脚本)
    - [配置config（可选）](#配置config)
    - [收集分数（可选）](#收集分数)
- [评测任务介绍](#评测任务介绍)

若需要进一步修改功能，更多细节详见[Document.md](Document.md)


## 开始运行

### 配置环境

主要需要torch、transformers、vllm库

```
pip install -r requirements.txt
```

### 下载数据
下载数据，并将数据文件data.zip解压到项目根目录

```
cd llm-cute-eval-main
wget https://github.com/PPMark0712/llm-cute-eval/releases/download/0.5/data.zip
unzip data.zip
```

### 运行参数

```
详细内容可查看./llm_cute_eval/run.py中的parse_args函数

模型配置
model_path: 模型绝对路径（也可以是hf路径）
model_type: 模型变量类型，默认vllm，目前可选vllm或hf
format_type: 模型类型，用于控制prompt格式，默认default，该参数目前仅建议使用default

任务配置
tasks: 需要评测的任务名称，用空格隔开，例如需要评测mmlu和gsm8k，则在命令中加入--tasks gsm8k mmlu，也可以包含一个all，自动评测所有任务。
config_path: 任务配置文件路径，可以为空，缺失值自动填充为对应任务文件夹中的默认config。
data_path(不需要使用): 数据集路径。

保存配置
output_path: 输出目录，默认./output
save_name: 输出文件夹名称。
save_infer_results: 保存推理结果，而非只保存一个分数
save_infer_texts: 保存便于阅读的输入输出文本到infer_result{round_idx}.txt
no_timestamp: 输出文件夹不包含时间戳。若包含时间戳，则会保存到"./output/{time}_{model_name}/"中
temp_file_path(不需要使用): 临时文件保存目录，主要用于humaneval评测集。

推理配置
tensor_parallel_size: 模型并行推理的设备数量，默认1。
rounds: 推理轮数，需要自己控制中间对话的prompt，具体见lm_cute_eval/get_multiround_prompt.py。
seed: 随机种子。
use_cpu(不需要使用): 使用CPU推理(用于debug)。
temperature: 模型温度
top_p: 模型推理采样参数
top_k: 模型推理采样参数
use_chat: 使用tokenizer.apply_chat_template，不建议使用，目前由于未知原因，使用该参数有时会导致模型无法正常回答
```

### 运行脚本

例如你想用mmlu和gsm8k评测两个模型：

```bash
export CUDA_VISIBLE_DEVICES=0  # 指定显卡编号
export TOKENIZERS_PARALLELISM=false  # 在humaneval评测中需要使用

declare -A models=(  # 批量评测模型
   	["model_name1"]="/path/to/model_1/"
	["model_name2"]="/path/to/model_2/"
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
        --config_path "" \
        --output_path output/debug \
        --tensor_parallel_size 1 \
        --temperature 0
done
```

### 多进程多卡运行脚本

若需要多卡数据并行加速，可以将运行命令改为多进程的版本，在单卡显存足够的情况下，多卡效率可能不如单卡。在评测多个模型时，可以将需要评测的模型分配到多个进程，每个进程用单卡评测一个模型。（此时多进程的进度条会有重叠，但是不影响结果）

```bash
export CUDA_VISIBLE_DEVICES=2,3
export TOKENIZERS_PARALLELISM=false
tensor_parallel_size=1

declare -A models=(
    ["model_name1"]="\path\to\model1"
    ["model_name2"]="\path\to\model2"
    # ...
)

output_path="\path\to\output"
tasks="task1 task2 ..."

# Calculate number of processes
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
num_gpus=${#GPUS[@]}
num_processes=$((num_gpus / tensor_parallel_size))
echo "Running with $num_processes processes"

# Get all model names into an array
model_names=("${!models[@]}")
total_models=${#model_names[@]}

# Function to run a subset of models
run_models() {
    local start_idx=$1
    local end_idx=$2
    local process_id=$3
    
    for ((i=start_idx; i<end_idx; i++)); do
        model_name="${model_names[$i]}"
        model_path=${models[$model_name]}
        echo "Process $process_id: Evaluating $model_name"
        
        # Set specific GPU for this process
        local gpu_id=${GPUS[$process_id]}
        CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
            --model_path "$model_path" \
            --model_type vllm \
            --format_type default \
            --tasks $tasks \
            --save_name "$model_name" \
            --save_infer_texts \
            --save_infer_results \
            --config_path "" \
            --output_path $output_path \
            --temperature 0.0 \
            --tensor_parallel_size $tensor_parallel_size \
            --no_timestamp
    done
}

# Distribute models across processes
models_per_process=$((total_models / num_processes))
remainder=$((total_models % num_processes))

start_idx=0
for ((p=0; p<num_processes; p++)); do
    # Calculate how many models this process should handle
    count=$models_per_process
    if ((p < remainder)); then
        ((count++))
    fi
    end_idx=$((start_idx + count))
    # Run this subset of models in background
    run_models $start_idx $end_idx $p &
    start_idx=$end_idx
done

# Wait for all processes to complete
wait
echo "All evaluations completed"
```

### 配置config

根目录下，有默认config.json文件，在每个数据集的config文件中也有默认值，可以根据需要来修改config中的内容，其格式如下：

```
{
    task_name_1: task_config_1,
    task_name_2: taks_config_2,
    ...
}

例如:
{
    "gsm8k": {
        "num_fewshots": 8,
        "limit": 0
    }
    "mmlu": {
        "num_fewshots": 5,
        "limit": null
    }
}
```

若某一项的config为空，或config中某个key为空，则会自动使用对应任务的模块中的config文件。由于每个任务需要的配置不同，所以需要自定义修改任务配置时，请查看./llm-cute-eval/tasks/<task_name>/config_<task_name>.json文件。


### 收集分数

可以运行/llm-cute-eval/gather_score.py脚本来收集分数，生成latex三线表格（输出到终端），其中每一列最高分加粗，第二名下划线。

```bash
python llm_cute_eval/gather_score.py \
    --dict_paths \
        /path/to/model_1/summary.json \
        /path/to/model_2/summary.json \
    --model_names \
        'model_name_1' \
        'model_name_2' \
    --tasks task_name_1 task_name_2 \
    --need_avg
```

## 评测任务介绍

### arc

数据集来源：使用[opencompass](https://github.com/open-compass/opencompass)中的数据。

评测指标：acc，匹配回答中的第一个选项，判断是否正确。

### cfinbench

数据集来源：使用[yanbinwei/CFinBench-Eval](https://github.com/yanbinwei/CFinBench-Eval)中的数据。

评测指标：acc。1、判断题：“正确”/“错误”出现在模型回答里。2、多选题：回答字符串中，句号前的回答和答案一样。3、单选题：匹配回答中的第一个选项。

### cmmlu

数据集来源：使用[opencompass](https://github.com/open-compass/opencompass)中的数据。

评测指标：匹配回答中的第一个选项，判断是否正确。

### commonsenseqa

数据集来源：使用[opencompass](https://github.com/open-compass/opencompass)中的数据。

评测指标：匹配回答中的第一个选项，判断是否正确。

### drop

数据集来源：

评测指标：正则表达式匹配例如'answer is'后面的回答，如果有一种可能的答案出现在回答中，则为正确。

### gsm8k

数据集来源：使用huggingface中的数据，fewshot prompt采用了[FranxYao/chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub)中的部分数据。

评测指标：

exact_match(acc)：匹配回答中'####'后面的数字，该数字正确则为正确。

flexible_match：匹配回答中任何数字，有一个正确则为正确。

### hellaswag

数据集来源：使用huggingface中的数据。

评测指标：匹配回答中的第一个选项，判断是否正确。

### humaneval

数据集来源：使用huggingface中的数据，不能设置fewshot，但是实际使用了2-shot来控制输出格式，便于提取代码块。

评测指标：找到第一个代码块，并用humaneval官方代码仓库[openai/human-eval](https://github.com/openai/human-eval)中的评测代码进行评测。

注意：该评测任务会不安全地执行模型生成的代码（尽管几乎不会出现严重事故），但是运行前请先仔细阅读/llm_cute_eval/tasks/humaneval/human_eval/execution.py中第48到57行的警告信息（来源humaneval官方代码仓库）。官方代码中，第58行被注释掉，要求用户自己取消注释才能运行，但在本项目中已经取消注释。

### icleval

数据集来源：[ICLEval/data/tasks_data](https://github.com/yiye3/ICLEval/tree/main/data/tasks_data)，修复了部分文件中"examples"写成"exmaples"的拼写错误。

评测指标：acc，若标准答案是回答的子串，则为正确。

### math

数据集来源：[HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)，fewshot使用了[FranxYao/chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub)中的fewshot_cot_prompt，匹配答案参考了[hendrycks/math](https://github.com/hendrycks/math)，但还加入了更多的latex字符串处理。

评测指标：acc

### mmlu

数据集来源：使用[opencompass](https://github.com/open-compass/opencompass)中的数据，以级[FranxYao/chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub)中的fewshot_cot_prompt。

评测指标：acc，匹配回答中的第一个选项，判断是否正确。

### mmluproplus（暂未完成）


评测指标：acc，匹配回答中的第一个选项，判断是否正确。

### winogrande

数据集来源：使用[opencompass](https://github.com/open-compass/opencompass)中的数据，用GPT4生成的fewshot_cot_prompt。

评测指标：acc，匹配回答中的第一个选项，判断是否正确。

### rgb

数据集来源：[chen700564/RGB](https://github.com/chen700564)

评测指标：acc，有一个可能的回答出现在回答里即为正确。

### xiezhi

数据集来源：[MikeGu721/XiezhiBenchmark](https://github.com/MikeGu721/XiezhiBenchmark/tree/main/Tasks/Knowledge/Benchmarks/test)

评测指标：acc，回答的编号正确或答案出现在回答里即为正确。

### xsum（暂未完成）

数据集来源：[EdinburghNLP/XSum(github.com)](https://github.com/EdinburghNLP/XSum)

评测指标：similarity，用BAAI/bge-m3模型计算和答案的相似度。