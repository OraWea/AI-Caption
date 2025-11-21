import os
import logging
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SubtitleDataset(Dataset):
    """
    自定义数据集类，用于加载字幕平行语料
    预期数据格式 JSON: [{"src": "Hello", "tgt": "你好"}, ...]
    """
    def __init__(self, data_path, tokenizer, max_length=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"正在加载数据集: {data_path}")
        try:
            # 在实际演示中，如果文件不存在，我们生成一些模拟数据以防报错
            if not os.path.exists(data_path):
                logger.warning(f"数据文件 {data_path} 不存在，生成模拟数据用于演示流程...")
                self.data = [
                    {"src": "Hello world", "tgt": "你好世界"},
                    {"src": "This is a test subtitle.", "tgt": "这是一个测试字幕。"},
                    {"src": "Machine learning is fascinating.", "tgt": "机器学习很迷人。"}
                ] * 50 # 复制多次以模拟数据量
            else:
                with open(data_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            
            logger.info(f"成功加载 {len(self.data)} 条数据")
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = item['src']
        tgt_text = item['tgt']

        # Tokenize 输入 (Source)
        model_inputs = self.tokenizer(
            src_text, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize 输出 (Target)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_text, 
                max_length=self.max_length, 
                padding="max_length", 
                truncation=True,
                return_tensors="pt"
            )

        # 将 input_ids 和 labels 展平 (remove batch dim 1)
        return {
            "input_ids": model_inputs["input_ids"][0],
            "attention_mask": model_inputs["attention_mask"][0],
            "labels": labels["input_ids"][0],
        }

class ModelFineTuner:
    """
    迁移学习控制器
    封装了完整的 NLLB 模型微调流程
    """
    def __init__(self, base_model_id="facebook/nllb-200-distilled-600M", output_dir="models/finetuned_v1"):
        self.base_model_id = base_model_id
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"初始化微调器 - 设备: {self.device}, 基座模型: {base_model_id}")

    def train(self, train_data_path, eval_data_path=None, epochs=3, batch_size=4):
        """
        执行训练循环
        """
        # 1. 加载 Tokenizer 和 模型
        logger.info("加载 Tokenizer 和 Model...")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model_id)
        model.to(self.device)

        # 2. 准备数据集
        train_dataset = SubtitleDataset(train_data_path, tokenizer)
        eval_dataset = SubtitleDataset(eval_data_path, tokenizer) if eval_data_path else None

        # 3. 定义数据整理器 (Data Collator)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        # 4. 设置训练参数
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="no", # 演示环境通常没有验证集，设为 no 防止报错
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=2,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(), # 如果有 GPU 则开启混合精度
            logging_steps=10,
            save_steps=100,
            report_to="none" # 不上传 wandb
        )

        # 5. 初始化 Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # 6. 开始训练
        logger.info(">>> 开始微调过程 (Transfer Learning) <<<")
        try:
            train_result = trainer.train()
            
            # 7. 保存模型
            logger.info(f"训练完成，正在保存模型至 {self.output_dir}...")
            trainer.save_model()
            tokenizer.save_pretrained(self.output_dir)
            
            # 保存训练指标
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            
            return True, metrics
            
        except Exception as e:
            logger.error(f"训练过程中断: {e}")
            return False, str(e)

if __name__ == "__main__":
    # 命令行测试入口
    print("=== 字幕模型迁移学习工具 ===")
    print("提示: 此脚本将下载 600M+ 模型文件并在本地微调")
    
    # 创建模拟数据文件以供测试
    dummy_data_path = "temp_train_data.json"
    if not os.path.exists(dummy_data_path):
        with open(dummy_data_path, 'w', encoding='utf-8') as f:
            json.dump([
                {"src": "Hello world", "tgt": "你好世界"}, 
                {"src": "AI subtitle generator", "tgt": "AI字幕生成器"}
            ], f)
    
    tuner = ModelFineTuner()
    tuner.train(train_data_path=dummy_data_path, epochs=1)
    
    # 清理测试文件
    if os.path.exists(dummy_data_path):
        os.remove(dummy_data_path)