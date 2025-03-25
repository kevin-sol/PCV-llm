# 导入必要的库
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType, get_peft_model_state_dict


# 定义不同模型类型的目标模块
# 这些模块将被应用LoRA低秩适配
TARGET_MODULES = {
    'llama': ["q_proj", "v_proj"],  # Llama模型的query和value投影
    'llava': ["q_proj", "v_proj"],  # LLaVA模型的query和value投影
    'mistral': ["q_proj", "v_proj"], # Mistral模型的query和value投影
    'opt': ["q_proj", "v_proj"],     # OPT模型的query和value投影
    'gpt2': ["q_proj", "v_proj"],    # GPT2模型的query和value投影
    't5-lm': ["q", "v"]              # T5模型的query和value投影
}


def print_trainable_parameters(model):
    """
    打印模型的可训练参数数量和总参数数量
    Args:
        model: 需要统计参数的模型
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()  # 统计所有参数数量
        if param.requires_grad:
            trainable_params += param.numel()  # 统计可训练参数数量
    
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def peft_model(plm, plm_type, rank, print_trainable=False, task_type=TaskType.FEATURE_EXTRACTION):
    """
    对预训练语言模型应用LoRA低秩适配
    Args:
        plm: 预训练语言模型
        plm_type: 模型类型,用于选择目标模块
        rank: LoRA的秩
        print_trainable: 是否打印可训练参数信息
        task_type: 任务类型,默认为特征提取
    Returns:
        添加了LoRA的模型
    """
    # 冻结所有参数,并将一维参数转换为float32类型
    for param in plm.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    # 启用梯度检查点和输入梯度
    plm.gradient_checkpointing_enable()
    plm.enable_input_require_grads()

    # 定义输出转换为float32的辅助类
    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    # 配置LoRA参数
    config = LoraConfig(
        r=rank,                                    # LoRA秩
        lora_alpha=32,                            # LoRA alpha缩放参数
        target_modules=TARGET_MODULES[plm_type],   # 目标模块
        lora_dropout=0.05,                        # LoRA dropout率
        bias="none",                              # 不使用偏置
        task_type=task_type                       # 任务类型
    )

    # 获取PEFT模型
    model = get_peft_model(plm, config)
    model.from_pretrained
    if print_trainable:
        print_trainable_parameters(model)
    return model
