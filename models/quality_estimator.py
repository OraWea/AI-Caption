import logging
import os

# 设置 Hugging Face 镜像，解决国内连接问题 (必须在导入 sentence_transformers 之前设置)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer, util
import torch

logger = logging.getLogger(__name__)

class QualityEstimator:
    def __init__(self, model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device='cpu'):
        """
        初始化质量评估模型
        Args:
            model_id: Hugging Face 模型 ID
            device: 运行设备 ('cpu' or 'cuda')
        """
        self.device = device
        self.model_id = model_id
        
        logger.info(f"正在加载 QE 模型 (Bi-Encoder): {model_id} (Device: {device})")
        try:
            self.model = SentenceTransformer(model_id, device=device)
            logger.info("QE 模型加载成功")
        except Exception as e:
            logger.error(f"QE 模型加载失败: {e}")
            raise e

    def estimate(self, source_text: str, translated_text: str) -> float:
        """
        评估翻译质量
        Args:
            source_text: 源语言文本
            translated_text: 翻译后的文本
        Returns:
            score: 0-1 之间的质量分数 (越高越好)
        """
        try:
            # Bi-Encoder: 分别编码，计算余弦相似度
            embeddings = self.model.encode([source_text, translated_text], convert_to_tensor=True)
            score = util.cos_sim(embeddings[0], embeddings[1])
            return float(score.item())
        except Exception as e:
            logger.error(f"QE 评估失败: {e}")
            return 0.0

    def estimate_batch(self, pairs: list) -> list:
        """
        批量评估翻译质量
        Args:
            pairs: list of [source_text, translated_text]
        Returns:
            scores: list of floats
        """
        try:
            # 拆分源文本和翻译文本
            sources = [p[0] for p in pairs]
            translations = [p[1] for p in pairs]
            
            # 批量编码
            src_embeddings = self.model.encode(sources, convert_to_tensor=True)
            trans_embeddings = self.model.encode(translations, convert_to_tensor=True)
            
            # 计算成对余弦相似度
            scores = util.cos_sim(src_embeddings, trans_embeddings)
            
            # 提取对角线元素 (即对应的相似度)
            # 注意：util.cos_sim 返回的是矩阵，我们需要一一对应的分数
            # 实际上 util.pairwise_cos_sim 更合适，或者手动提取
            # 这里简单起见，使用对角线
            
            # 修正：SentenceTransformers 没有 pairwise_cos_sim，需要手动计算
            # 或者简单循环 (对于小批量没问题)
            
            batch_scores = []
            for i in range(len(pairs)):
                # 提取对应向量的点积 (归一化后即余弦相似度)
                s = util.cos_sim(src_embeddings[i], trans_embeddings[i])
                batch_scores.append(float(s.item()))
                
            return batch_scores
        except Exception as e:
            logger.error(f"批量 QE 评估失败: {e}")
            return [0.0] * len(pairs)
