import torch
import os


# 保存模型函数
def save_model(args, model, save_dir):
    if args.rank > 0:
        # 如果使用低秩矩阵,分别保存lora权重和其他模块
        model.plm.save_pretrained(save_dir)
        # save other modules except plm
        torch.save(model.modules_except_plm.state_dict(), os.path.join(save_dir, 'modules_except_plm.bin'))
    else:
        # 如果不使用低秩矩阵,保存整个模型
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.bin'))

# 加载模型函数
def load_model(args, model, model_dir):
    if args.rank > 0:
        # 如果使用低秩矩阵,分别加载lora权重和其他模块
        model.plm.load_adapter(model_dir, adapter_name='default')
        # load other modules except plm
        model.modules_except_plm.load_state_dict(torch.load(os.path.join(model_dir, 'modules_except_plm.bin')))
    else:
        # 如果不使用低秩矩阵,加载整个模型
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.bin')))
    return model
