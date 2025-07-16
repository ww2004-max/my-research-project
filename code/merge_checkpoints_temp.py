import os, glob, sys

sys.path.append('C:/Users/33491/PycharmProjects/machine/fairseq')
checkpoint_dir = 'C:/Users/33491/PycharmProjects/machine/pdec_work/checkpoints/ted_pdec_mini/1'
output_file = 'C:/Users/33491/PycharmProjects/machine/pdec_work/checkpoints/ted_pdec_mini/1/checkpoint_averaged.pt'

print(f'���Ҽ���Ŀ¼: {checkpoint_dir}')

# ��������best_loss����
checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint.best_loss_*')
checkpoints = glob.glob(checkpoint_pattern)

if not checkpoints:
    print(f' ���棺�� {checkpoint_dir} ��δ�ҵ��κ� best_loss ����')
    
    # ���Բ�����������
    print('���Բ�����������...')
    all_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint*'))
    if all_checkpoints:
        print(f'�ҵ����¼��㣺')
        for cp in all_checkpoints:
            print(f'  - {os.path.basename(cp)}')
    else:
        print(f' ���棺�� {checkpoint_dir} ��δ�ҵ��κμ���')
else:
    print(f'�ҵ����¼��㣺')
    for cp in checkpoints:
        print(f'  - {os.path.basename(cp)}')
    
    # ����fairseq��average_checkpoints�ű�
    try:
        from fairseq.scripts.average_checkpoints import main as average_checkpoints_main
        sys.argv = ['average_checkpoints.py', '--inputs'] + checkpoints + ['--output', output_file]
        print(f'���ںϲ����㵽��{output_file}')
        average_checkpoints_main()
        print('����ϲ���ɣ�')
    except Exception as e:
        print(f'�ϲ�����ʱ����: {e}')
