#!/usr/bin/env python
"""
Windows环境下的检查点合并工具
"""

import os
import glob
import sys
import traceback

# 添加fairseq目录到Python路径
fairseq_path = "C:/Users/33491/PycharmProjects/machine/fairseqirseq"
sys.path.append(fairseq_path)

print(f"Python路径: {sys.path}")
print(f"尝试导入 fairseq.scripts.average_checkpoints...")

try:
    # 尝试直接导入脚本
    sys.path.append(os.path.join(fairseq_path, "scripts"))
    print(f"检查scripts目录是否存在: {os.path.exists(os.path.join(fairseq_path, 'scripts'))}")

    try:
        from fairseq.scripts.average_checkpoints import main as average_checkpoints_main

        print("成功导入 fairseq.scripts.average_checkpoints!")
    except ImportError as e:
        print(f"从fairseq.scripts导入失败: {e}")
        print("尝试直接导入脚本...")

        try:
            sys.path.append(os.path.join(fairseq_path, "scripts"))
            from average_checkpoints import main as average_checkpoints_main

            print("成功从scripts目录直接导入!")
        except ImportError as e2:
            print(f"直接导入也失败了: {e2}")
            print("检查average_checkpoints.py是否存在:")
            script_path = os.path.join(fairseq_path, "scripts", "average_checkpoints.py")
            print(f"文件存在: {os.path.exists(script_path)}")

            if os.path.exists(script_path):
                print("尝试手动执行脚本...")
                # 如果文件存在，我们可以尝试手动执行它
                import importlib.util

                spec = importlib.util.spec_from_file_location("average_checkpoints", script_path)
                average_checkpoints = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(average_checkpoints)
                average_checkpoints_main = average_checkpoints.main
                print("成功手动导入脚本!")
            else:
                print("错误：无法找到average_checkpoints.py文件")
                sys.exit(1)
except Exception as e:
    print(f"发生未预期的错误: {e}")
    print(traceback.format_exc())
    sys.exit(1)


def main():
    try:
        checkpoint_dir = "C:/Users/33491/PycharmProjects/machine/pdec_work/checkpoints/ted_pdec_mini/1"
        output_file = "C:/Users/33491/PycharmProjects/machine/pdec_work/checkpoints/ted_pdec_mini/1/checkpoint_averaged.pt"

        print(f"查找检查点目录: {checkpoint_dir}")
        print(f"目录存在: {os.path.exists(checkpoint_dir)}")

        # 查找所有best_loss检查点
        checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint.best_loss_*.pt")
        checkpoints = glob.glob(checkpoint_pattern)

        if not checkpoints:
            print(f"⚠️ 警告：在 {checkpoint_dir} 中未找到任何 best_loss 检查点")

            # 尝试查找其他检查点
            print("尝试查找其他检查点...")
            all_checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint*.pt"))
            if all_checkpoints:
                print(f"找到以下检查点：")
                for cp in all_checkpoints:
                    print(f"  - {os.path.basename(cp)}")

                # 使用checkpoint_best.pt和checkpoint_last.pt
                best_cp = os.path.join(checkpoint_dir, "checkpoint_best.pt")
                last_cp = os.path.join(checkpoint_dir, "checkpoint_last.pt")
                selected_checkpoints = []

                if os.path.exists(best_cp):
                    selected_checkpoints.append(best_cp)
                    print(f"将使用: {os.path.basename(best_cp)}")

                if os.path.exists(last_cp):
                    selected_checkpoints.append(last_cp)
                    print(f"将使用: {os.path.basename(last_cp)}")

                if selected_checkpoints:
                    # 调用fairseq的average_checkpoints脚本
                    sys.argv = ["average_checkpoints.py", "--inputs"] + selected_checkpoints + ["--output", output_file]
                    print(f"正在合并检查点到：{output_file}")
                    print(f"执行命令: {sys.argv}")
                    average_checkpoints_main()
                    print("检查点合并完成！")
                else:
                    print("没有找到可用的检查点进行合并")
            else:
                print(f"⚠️ 警告：在 {checkpoint_dir} 中未找到任何检查点")
        else:
            print(f"找到以下best_loss检查点：")
            for cp in checkpoints:
                print(f"  - {os.path.basename(cp)}")

            # 调用fairseq的average_checkpoints脚本
            sys.argv = ["average_checkpoints.py", "--inputs"] + checkpoints + ["--output", output_file]
            print(f"正在合并检查点到：{output_file}")
            print(f"执行命令: {sys.argv}")
            average_checkpoints_main()
            print("检查点合并完成！")
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()