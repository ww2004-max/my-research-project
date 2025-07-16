#!/usr/bin/env python3
import os

def fix_utils_file():
    """全面修复utils.py文件中的缩进问题"""
    utils_file = os.path.join('fairseq', 'fairseq', 'utils.py')
    
    print(f"开始修复 {utils_file}...")
    
    # 读取文件内容
    with open(utils_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份原文件
    backup_file = utils_file + '.bak_full'
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"原文件已备份为 {backup_file}")
    
    # 替换有问题的代码块
    problematic_code = """    if getattr(args, "user_dir", None) is None or args.user_dir == "":
        return
        
    module_path = args.user_dir

    # check if --user-dir is a file, in which case the module name is the
    # filename without the extension
        if os.path.isfile(module_path):
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        module_path = os.path.dirname(module_path)
        if module_path != "":
            sys.path.insert(0, module_path)
            else:
            sys.path.insert(0, ".")
                else:
        last_path_component = os.path.basename(module_path)
        possible_module_name = last_path_component
        # if the last part of the path is "fairseq" or "fairseq-py", then we use
        # the previous part as the name
        if possible_module_name == "fairseq" or possible_module_name == "fairseq-py":
            possible_module_name = os.path.basename(os.path.dirname(module_path))

        # if we find fairseq/xla or fairseq/fb/xla, we know that the actual
        # module is fairseq.xla or fairseq.fb.xla
        if os.path.exists(os.path.join(module_path, "fairseq", "xla")):
            module_name = "fairseq.xla"
            sys.path.insert(0, module_path)
        elif os.path.exists(os.path.join(module_path, "fairseq", "fb", "xla")):
            module_name = "fairseq.fb.xla"
            sys.path.insert(0, module_path)
        else:
            module_name = possible_module_name
            # the path could be an absolute path to a file in the directory
    if os.path.isfile(module_path):
                module_path = os.path.dirname(module_path)
            sys.path.insert(0, module_path)"""
    
    fixed_code = """    if getattr(args, "user_dir", None) is None or args.user_dir == "":
        return
        
    module_path = args.user_dir

    # check if --user-dir is a file, in which case the module name is the
    # filename without the extension
    if os.path.isfile(module_path):
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        module_path = os.path.dirname(module_path)
        if module_path != "":
            sys.path.insert(0, module_path)
        else:
            sys.path.insert(0, ".")
    else:
        last_path_component = os.path.basename(module_path)
        possible_module_name = last_path_component
        # if the last part of the path is "fairseq" or "fairseq-py", then we use
        # the previous part as the name
        if possible_module_name == "fairseq" or possible_module_name == "fairseq-py":
            possible_module_name = os.path.basename(os.path.dirname(module_path))

        # if we find fairseq/xla or fairseq/fb/xla, we know that the actual
        # module is fairseq.xla or fairseq.fb.xla
        if os.path.exists(os.path.join(module_path, "fairseq", "xla")):
            module_name = "fairseq.xla"
            sys.path.insert(0, module_path)
        elif os.path.exists(os.path.join(module_path, "fairseq", "fb", "xla")):
            module_name = "fairseq.fb.xla"
            sys.path.insert(0, module_path)
        else:
            module_name = possible_module_name
            # the path could be an absolute path to a file in the directory
            if os.path.isfile(module_path):
                module_path = os.path.dirname(module_path)
            sys.path.insert(0, module_path)"""
    
    # 替换内容
    new_content = content.replace(problematic_code, fixed_code)
    
    # 写入修复后的内容
    with open(utils_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("修复完成!")

if __name__ == "__main__":
    fix_utils_file() 