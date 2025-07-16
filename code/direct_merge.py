#!/usr/bin/env python3
# 简化版的检查点合并脚本，不依赖fairseq.file_io

import argparse
import collections
import os
import sys
import torch

def average_checkpoints(inputs):
    """加载检查点并返回具有平均权重的模型。"""
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    
    print(f"合并 {num_models} 个检查点...")
    
    for i, fpath in enumerate(inputs):
        print(f"处理检查点 {i+1}/{num_models}: {fpath}")
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
            )
        
        # 从第一个检查点复制设置
        if new_state is None:
            new_state = state
        
        model_params = state["model"]
        model_params_keys = list(model_params.keys())
        
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                f"检查点 {fpath} 的参数列表与预期不符"
            )
        
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
            else:
                params_dict[k] += p
    
    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    
    new_state["model"] = averaged_params
    return new_state

def main():
    # 设置参数
    checkpoint_dir = r"C:\Users\33491\PycharmProjects\machine\pdec_work\checkpoints\ted_pdec_mini\1"
    output_file = os.path.join(checkpoint_dir, "checkpoint_averaged.pt")
    
    # 查找最佳检查点
    inputs = [os.path.join(checkpoint_dir, "checkpoint.best_loss_6.9432.pt")]
    
    if not inputs:
        print("⚠️ 警告：未找到检查点文件")
        return
    
    print(f"找到 {len(inputs)} 个检查点文件")
    for inp in inputs:
        print(f"- {inp}")
    
    # 合并检查点
    new_state = average_checkpoints(inputs)
    
    # 保存合并后的检查点
    print(f"保存合并后的检查点到 {output_file}")
    torch.save(new_state, output_file)
    print(f"✅ 完成! 合并后的检查点已保存到 {output_file}")

if __name__ == "__main__":
    main() 