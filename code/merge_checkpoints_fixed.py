import os
import glob
import sys
import subprocess

# 添加fairseq到Python路径
sys.path.append('C:/Users/33491/PycharmProjects/machine/fairseq')

# 设置检查点目录和输出文件路径
checkpoint_dir = 'C:/Users/33491/PycharmProjects/machine/pdec_work/checkpoints/ted_pdec_mini/1'
output_file = 'C:/Users/33491/PycharmProjects/machine/pdec_work/checkpoints/ted_pdec_mini/1/checkpoint_averaged.pt'

print(f'正在查找检查点目录: {checkpoint_dir}')

# 查找所有best_loss检查点
checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint.best_loss_*')
checkpoints = glob.glob(checkpoint_pattern)

if not checkpoints:
    print(f'警告：在 {checkpoint_dir} 中未找到任何 best_loss 检查点')
    
    # 尝试查找其他检查点
    print('尝试查找其他检查点...')
    all_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint*'))
    if all_checkpoints:
        print(f'找到以下检查点：')
        for cp in all_checkpoints:
            print(f'  - {os.path.basename(cp)}')
    else:
        print(f'警告：在 {checkpoint_dir} 中未找到任何检查点')
else:
    print(f'找到以下检查点：')
    for cp in checkpoints:
        print(f'  - {os.path.basename(cp)}')
    
    # 使用fairseq的average_checkpoints.py脚本
    try:
        # 构建命令行参数
        cmd = [
            sys.executable,  # 当前Python解释器
            "-m", "fairseq.scripts.average_checkpoints",
            "--inputs"
        ] + checkpoints + [
            "--output", output_file
        ]
        
        print(f'执行命令: {" ".join(cmd)}')
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print('检查点合并完成！')
            print(result.stdout)
        else:
            print(f'合并检查点时出错:')
            print(result.stderr)
            
            # 尝试替代方法
            print('尝试使用替代方法合并检查点...')
            # 简单的模型参数平均
            import torch
            
            # 加载所有检查点
            models = [torch.load(cp, map_location='cpu') for cp in checkpoints]
            
            # 创建平均模型
            avg_model = {}
            for key in models[0].keys():
                if key == 'model':
                    avg_model[key] = {}
                    for param_key in models[0][key].keys():
                        avg_model[key][param_key] = sum(model[key][param_key] for model in models) / len(models)
                else:
                    avg_model[key] = models[0][key]
            
            # 保存平均模型
            torch.save(avg_model, output_file)
            print(f'使用替代方法合并检查点完成，已保存到 {output_file}')
            
    except Exception as e:
        print(f'合并检查点时出错: {e}') 