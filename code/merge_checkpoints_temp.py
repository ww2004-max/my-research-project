import os, glob, sys

sys.path.append('C:/Users/33491/PycharmProjects/machine/fairseq')
checkpoint_dir = 'C:/Users/33491/PycharmProjects/machine/pdec_work/checkpoints/ted_pdec_mini/1'
output_file = 'C:/Users/33491/PycharmProjects/machine/pdec_work/checkpoints/ted_pdec_mini/1/checkpoint_averaged.pt'

print(f'查找检查点目录: {checkpoint_dir}')

# 查找所有best_loss检查点
checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint.best_loss_*')
checkpoints = glob.glob(checkpoint_pattern)

if not checkpoints:
    print(f' 警告：在 {checkpoint_dir} 中未找到任何 best_loss 检查点')
    
    # 尝试查找其他检查点
    print('尝试查找其他检查点...')
    all_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint*'))
    if all_checkpoints:
        print(f'找到以下检查点：')
        for cp in all_checkpoints:
            print(f'  - {os.path.basename(cp)}')
    else:
        print(f' 警告：在 {checkpoint_dir} 中未找到任何检查点')
else:
    print(f'找到以下检查点：')
    for cp in checkpoints:
        print(f'  - {os.path.basename(cp)}')
    
    # 调用fairseq的average_checkpoints脚本
    try:
        from fairseq.scripts.average_checkpoints import main as average_checkpoints_main
        sys.argv = ['average_checkpoints.py', '--inputs'] + checkpoints + ['--output', output_file]
        print(f'正在合并检查点到：{output_file}')
        average_checkpoints_main()
        print('检查点合并完成！')
    except Exception as e:
        print(f'合并检查点时出错: {e}')
