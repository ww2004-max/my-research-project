from comet import download_model, load_from_checkpoint
import sys
import os

# 设置代理环境变量
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'

language_sequence = ["en", "ar", "he", "ru", "ko", "it", "ja", "zh", "es", "nl", "vi", "tr", "pl", "ro", "fa", "hr", "cs", "de"]

def _read_txt_strip_(url):
    file = open(url, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    return [line.strip() for line in lines]

method_name = sys.argv[1]
model_id = sys.argv[2]
tgt  = sys.argv[3]
save_path = sys.argv[4]
save_path = os.path.join(save_path, "results")

# 尝试使用本地模型，如果不存在则使用默认模型
local_model_path = os.path.join(save_path, "models", "wmt22-comet-da")
if os.path.exists(local_model_path):
    print(f"使用本地模型: {local_model_path}")
    model_path = local_model_path
else:
    print("尝试下载模型，如果失败将使用默认模型...")
    try:
        model_path = download_model("Unbabel/wmt22-comet-da")
    except Exception as e:
        print(f"下载模型失败: {e}")
        # 使用默认模型
        model_path = "wmt20-comet-da"
        print(f"使用默认模型: {model_path}")

try:
    print(f"加载模型: {model_path}")
    model = load_from_checkpoint(model_path)

    writing_list = []
    for src in language_sequence:
        if src == tgt: continue
        refs_path = os.path.join(save_path, str(method_name), str(model_id), "{}-{}.detok.r".format(src, tgt))
        hypos_path = os.path.join(save_path, str(method_name), str(model_id), "{}-{}.detok.h".format(src, tgt))
        srcs_path = os.path.join(save_path, str(method_name), str(model_id), "{}-{}.detok.s".format(src, tgt))
        
        if not os.path.exists(refs_path) or not os.path.exists(hypos_path) or not os.path.exists(srcs_path):
            print(f"跳过 {src}-{tgt}，文件不存在")
            continue
            
        print(f"处理 {src}-{tgt}...")
        refs = _read_txt_strip_(refs_path)
        hypos = _read_txt_strip_(hypos_path)
        srcs = _read_txt_strip_(srcs_path)
        data = [{"src": src_text, "mt": mt_text, "ref": ref_text} for src_text, mt_text, ref_text in zip(srcs, hypos, refs)]
        model_output = model.predict(data, batch_size=8, gpus=0)  # 减小batch size，不使用GPU
        score = round(model_output.system_score * 100, 2)
        writing_list.append(f"{src}-{tgt}\n")
        writing_list.append(f"Score: {score} \n")

    output_file = os.path.join(save_path, str(method_name), str(model_id), "{}.comet".format(str(model_id)))
    print(f"写入结果到: {output_file}")
    file = open(output_file, 'a', encoding='utf-8')
    file.writelines(writing_list)
    file.close()
    print("评估完成!")
except Exception as e:
    print(f"评估过程中出错: {e}")
