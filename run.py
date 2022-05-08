from model import DualSC
from utils import set_seed

set_seed()
# 初始化模型

model = DualSC(config_path = 'Config', beam_size = 3, max_source_length = 64, max_target_length = 64, load_model_path = None)

# 模型训练
model.train(train_filename ='data/train.csv', train_batch_size = 16, num_train_epochs = 10, learning_rate = 2e-4,
            do_eval = True, dev_filename ='data/valid.csv', eval_batch_size = 16, output_dir ='valid_output/')
#
# 模型测试
model = DualSC(config_path = 'Config', beam_size = 3, max_source_length = 64, max_target_length = 64, load_model_path = 'valid_output/checkpoint-best-bleu/pytorch_model.bin')

model.test(test_filename ='data/test.csv', test_batch_size = 16, output_dir ='test_output/')

# 模型推理
# comment = model.predict(source = 'ShellCodeSum: push eax,push eax onto the stack')
# print(comment)