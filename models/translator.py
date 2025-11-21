import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import logging
import gc

from models.quality_estimator import QualityEstimator
from config import Config

logger = logging.getLogger(__name__)

class NeuralTranslator:
    def __init__(self, nmt_model_id="facebook/nllb-200-distilled-600M", reflection_model_id=None, device='cpu'):
        self.device = device
        self.nmt_model_id = nmt_model_id
        
        logger.info(f"正在加载 NMT 模型: {nmt_model_id} (Device: {device})")
        try:
            # 1. 加载翻译模型 (NLLB)
            self.nmt_tokenizer = AutoTokenizer.from_pretrained(nmt_model_id)
            self.nmt_model = AutoModelForSeq2SeqLM.from_pretrained(nmt_model_id)
            
            if device == 'cuda':
                self.nmt_model = self.nmt_model.to(device)
            
            # 2. 加载反思模型 (可选)
            self.reflector = None
            if reflection_model_id:
                logger.info(f"正在加载反思模型: {reflection_model_id}")
                try:
                    # 使用 pipeline 简化 LLM 推理
                    # device_map="auto" 会自动处理内存
                    self.reflector = pipeline(
                        "text-generation", 
                        model=reflection_model_id,
                        device=0 if device == 'cuda' else -1,
                        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                        model_kwargs={"low_cpu_mem_usage": True}
                    )
                    logger.info("反思模型加载成功")
                except Exception as e:
                    logger.warning(f"反思模型加载失败，将降级为仅翻译模式: {e}")
                    self.reflector = None

            # 3. 加载 QE 模型 (可选)
            self.qe_model = None
            self.qe_threshold = getattr(Config, 'QE_THRESHOLD', 0.7)
            if getattr(Config, 'ENABLE_QE', False):
                qe_model_id = getattr(Config, 'QE_MODEL_ID', "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                try:
                    self.qe_model = QualityEstimator(model_id=qe_model_id, device=device)
                except Exception as e:
                    logger.warning(f"QE 模型加载失败: {e}")
                    self.qe_model = None
                
        except Exception as e:
            logger.error(f"核心模型加载失败: {e}")
            raise e

    def _translate_batch(self, texts, src_lang_code, tgt_lang_code):
        """
        执行批量翻译
        """
        # NLLB 语言代码映射表 (简化版，覆盖常见语言)
        # 完整列表见 NLLB 文档
        lang_map = {
            'en': 'eng_Latn', 
            'zh': 'zho_Hans', 'zh-cn': 'zho_Hans',
            'ja': 'jpn_Jpan', 
            'ko': 'kor_Hang',
            'fr': 'fra_Latn',
            'de': 'deu_Latn',
            'auto': 'eng_Latn' # 默认
        }
        
        # 获取 NLLB 特定的语言代码
        tgt_code = lang_map.get(tgt_lang_code, 'eng_Latn')
        src_code = lang_map.get(src_lang_code, 'eng_Latn')

        # 1. Tokenize
        inputs = self.nmt_tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        
        if self.device == 'cuda':
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2. Generate (Inference)
        # forced_bos_token_id 告诉模型翻译成什么语言
        forced_bos_token_id = self.nmt_tokenizer.convert_tokens_to_ids(tgt_code)
        
        with torch.no_grad():
            generated_tokens = self.nmt_model.generate(
                **inputs, 
                forced_bos_token_id=forced_bos_token_id,
                max_length=128,
                num_beams=4, # 使用 Beam Search 提升质量
                early_stopping=True
            )

        # 3. Decode
        translations = self.nmt_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # 清理显存
        if self.device == 'cuda':
            del inputs, generated_tokens
            torch.cuda.empty_cache()
            
        return translations

    def _reflect_and_improve(self, source_text, initial_translation, tgt_lang):
        """
        Agent 逻辑: 翻译 - 反思 - 改进
        """
        if not self.reflector:
            return initial_translation

        # 构建 Prompt (针对小模型优化)
        prompt = (
            f"Task: Correct the translation.\n"
            f"Original: {source_text}\n"
            f"Draft: {initial_translation}\n"
            f"Corrected: "
        )
        
        try:
            # LLM 推理
            response = self.reflector(
                prompt, 
                max_new_tokens=64, 
                do_sample=False, # 确定性输出
                num_return_sequences=1,
                pad_token_id=self.reflector.tokenizer.eos_token_id
            )[0]['generated_text']
            
            # 提取结果 (截取 Corrected: 之后的部分)
            improved = response.split("Corrected:")[-1].strip()
            
            # 简单的质量控制：如果生成结果太短或为空，回退
            if len(improved) < 2:
                return initial_translation
                
            return improved
        except Exception as e:
            logger.warning(f"反思步骤出错: {e}")
            return initial_translation

    def translate_segments(self, segments, target_lang, source_lang='auto', use_reflection=False):
        """
        处理字幕片段列表
        """
        translated_segments = []
        batch_size = 8 # 根据显存大小调整
        
        # 提取所有文本
        all_texts = [seg['text'] for seg in segments]
        
        # 批量翻译
        logger.info(f"开始批量翻译 {len(all_texts)} 条字幕...")
        final_translations = []
        
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i+batch_size]
            # Step 1: NMT 翻译
            batch_results = self._translate_batch(batch_texts, source_lang, target_lang)
            final_translations.extend(batch_results)
            
        # 批量 QE 评估 (如果启用)
        qe_scores = []
        if self.qe_model:
            logger.info("正在执行批量质量评估 (QE)...")
            pairs = [[seg['text'], trans] for seg, trans in zip(segments, final_translations)]
            qe_scores = self.qe_model.estimate_batch(pairs)
        else:
            qe_scores = [0.0] * len(segments)

        # 组装结果
        for i, seg in enumerate(segments):
            initial_trans = final_translations[i]
            final_text = initial_trans
            qe_score = qe_scores[i]
            
            # Step 2: 反思 (逐条处理，因为反思很慢)
            # 只有当启用反思 AND (没有QE模型 OR QE分数低于阈值) 时才反思
            should_reflect = use_reflection and self.reflector and len(initial_trans) > 5
            
            if should_reflect:
                if self.qe_model and qe_score >= self.qe_threshold:
                    logger.info(f"QE ({qe_score:.4f}) >= {self.qe_threshold}, 跳过反思: {initial_trans[:30]}...")
                else:
                    if self.qe_model:
                        logger.info(f"QE ({qe_score:.4f}) < {self.qe_threshold}, 触发反思: {initial_trans[:30]}...")
                    final_text = self._reflect_and_improve(seg['text'], initial_trans, target_lang)
                
            translated_segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': final_text,
                'original_text': seg['text'],
                'qe_score': qe_score # 记录分数以便调试
            })
            
        return translated_segments

    def get_supported_languages(self):
        return {
            "whisper": ["auto", "en", "zh", "ja", "ko", "fr", "de", "es"],
            "nmt": ["zh-cn", "en", "ja", "ko", "fr", "de", "es"]
        }