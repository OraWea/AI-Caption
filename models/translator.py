import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import logging
import gc
from typing import Optional, Dict, Any, List
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

class NeuralTranslator:
    def __init__(self, nmt_model_id="facebook/nllb-200-distilled-600M", reflection_model_id=None, device='cpu'):
        self.device = device
        self.nmt_tokenizer = None
        self.nmt_model = None
        self.reflector = None
        self.qe_model = None
        self.qe_threshold = 0.7  # 默认 QE 阈值（0-1，越高表示翻译质量越好）

        self._load_models(nmt_model_id, reflection_model_id)

    def _load_models(self, nmt_model_id: str, reflection_model_id: Optional[str]):
        """加载 NMT 模型、反思模型和 QE 模型"""
        try:
            # 1. 加载 NMT 模型（NLLB，支持多语言翻译）
            logger.info(f"Loading NMT model: {nmt_model_id} (Device: {self.device})")
            self.nmt_tokenizer = AutoTokenizer.from_pretrained(nmt_model_id)
            self.nmt_model = AutoModelForSeq2SeqLM.from_pretrained(
                nmt_model_id,
                use_safetensors=True,
                low_cpu_mem_usage=True
            ).to(self.device)
            logger.info("✅ NMT model loaded successfully.")

            # 2. 加载反思模型（可选，用于优化翻译结果）
            if reflection_model_id:
                logger.info(f"Loading reflection model: {reflection_model_id}")
                self.reflector = pipeline(
                    "text-generation",
                    model=reflection_model_id,
                    device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
                    dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    model_kwargs={"low_cpu_mem_usage": True}
                )
                # 处理反思模型的 pad token（若未设置）
                if self.reflector.tokenizer.pad_token is None:
                    self.reflector.tokenizer.pad_token = self.reflector.tokenizer.eos_token
                    logger.warning(f"Set pad_token to eos_token for reflection model.")
                logger.info("✅ Reflection model loaded successfully.")
            else:
                logger.warning("No reflection model specified, skipping optimization.")

            # 3. 加载 QE 模型（句子相似度模型，用于评估翻译质量）
            self._load_qe_model()

        except Exception as e:
            logger.error(f"Model load failed: {e}", exc_info=True)
            self._cleanup_vram()
            raise Exception(f"Translator init error: {str(e)}")

    def _load_qe_model(self):
        """加载翻译质量评估（QE）模型"""
        try:
            self.qe_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
            logger.info("✅ QE model (sentence-transformers) loaded successfully.")
        except Exception as e:
            logger.warning(f"QE model load failed: {e}", exc_info=True)
            self.qe_model = None

    def translate_segments(self, segments: List[Dict[str, Any]], target_lang: str, source_lang: str = 'auto',
                           use_reflection: bool = False, av_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        翻译字幕片段（支持片段级 AV 上下文优化）
        :param segments: 字幕片段列表（每个片段需包含 "start", "end", "text", "av_context"）
        :param target_lang: 目标语言代码（如 "zh", "en"）
        :param source_lang: 源语言代码（默认 "auto" 自动检测）
        :param use_reflection: 是否启用反思优化（默认 False）
        :param av_context: 全局 AV 上下文（优先级低于片段自身的 av_context）
        :return: 翻译后的字幕片段列表（含翻译文本、QE 分数、场景信息）
        """
        # 校验输入片段是否包含必要字段
        for idx, seg in enumerate(segments):
            if not all(key in seg for key in ["start", "end", "text"]):
                raise ValueError(f"Segment {idx+1} missing required fields (start/end/text).")
            # 为无 av_context 的片段添加默认值
            if "av_context" not in seg:
                seg["av_context"] = av_context or {}
                logger.warning(f"Segment {idx+1} has no av_context, using global context.")

        # 批量提取源文本（用于批量翻译，提升效率）
        source_texts = [seg["text"].strip() for seg in segments]
        logger.info(f"Starting translation: {len(source_texts)} segments -> Target lang: {target_lang} (Reflection: {use_reflection})")

        # 第一步：批量翻译（基础翻译结果）
        translated_texts = self._translate_batch(source_texts, source_lang, target_lang)
        logger.info(f"Batch translation completed.")

        # 第二步：反思优化（逐片段结合自身 AV 上下文优化）
        if use_reflection and self.reflector:
            logger.info("Starting reflection optimization with segment-level AV context...")
            optimized_texts = []
            for idx, (seg, src_text, trans_text) in enumerate(zip(segments, source_texts, translated_texts)):
                # 提取当前片段的 AV 上下文（优先级：片段自身 > 全局）
                segment_av_ctx = seg["av_context"] or av_context or {}
                # 优化当前片段的翻译结果
                optimized = self._reflect_and_improve(src_text, trans_text, target_lang, segment_av_ctx, idx+1)
                optimized_texts.append(optimized)
            translated_texts = optimized_texts
            logger.info("Reflection optimization completed.")

        # 第三步：计算 QE 分数（评估翻译质量）
        qe_scores = self._calculate_batch_qe_scores(source_texts, translated_texts) if self.qe_model else [0.0]*len(source_texts)

        # 第四步：组装最终结果（保留原始信息和场景信息）
        result = []
        for idx, (seg, trans_text, qe_score) in enumerate(zip(segments, translated_texts, qe_scores)):
            result.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": trans_text,
                "original_text": seg["text"],
                "qe_score": round(qe_score, 2),  # 翻译质量分数（0-1）
                "av_context": seg["av_context"],  # 保留当前片段的 AV 上下文
                "is_optimized": use_reflection and self.reflector is not None  # 是否经过反思优化
            })

        logger.info(f"Translation process finished: {len(result)} segments processed.")
        return result

    def _translate_batch(self, texts: List[str], src_lang_code: str, tgt_lang_code: str) -> List[str]:
        """
        批量翻译文本（NLLB 模型核心翻译逻辑）
        :param texts: 源文本列表
        :param src_lang_code: 源语言代码
        :param tgt_lang_code: 目标语言代码
        :return: 翻译后的文本列表
        """
        # NLLB 语言代码映射（扩展常见语言，可根据需求补充）
        lang_map = {
            'auto': 'eng_Latn', 'en': 'eng_Latn', 'zh': 'zho_Hans', 'zh-cn': 'zho_Hans',
            'ja': 'jpn_Jpan', 'ko': 'kor_Hang', 'fr': 'fra_Latn', 'de': 'deu_Latn',
            'es': 'spa_Latn', 'ru': 'rus_Cyrl', 'ar': 'ara_Arab', 'hi': 'hin_Deva',
            'pt': 'por_Latn', 'it': 'ita_Latn', 'nl': 'nld_Latn', 'pl': 'pol_Latn'
        }

        # 解析语言代码（源语言默认英语，目标语言默认中文）
        src_code = lang_map.get(src_lang_code.lower(), 'eng_Latn')
        tgt_code = lang_map.get(tgt_lang_code.lower(), 'zho_Hans')
        logger.debug(f"Batch translation: src_code={src_code}, tgt_code={tgt_code}, text_count={len(texts)}")

        # 预处理输入文本（tokenize）
        inputs = self.nmt_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128  # 限制最大长度，避免显存溢出
        ).to(self.device)

        # 强制目标语言 BOS token（NLLB 模型特殊要求）
        forced_bos_token_id = self.nmt_tokenizer.convert_tokens_to_ids(tgt_code)

        # 生成翻译（批量处理，提升效率）
        with torch.no_grad():
            generated_tokens = self.nmt_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=150,  # 翻译结果最大长度
                num_beams=4,     # 束搜索宽度（平衡质量和速度）
                early_stopping=True,
                no_repeat_ngram_size=2  # 避免重复短语
            )

        # 解码翻译结果（去除特殊 token）
        translations = self.nmt_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        # 清理首尾空格
        translations = [trans.strip() for trans in translations]

        return translations

    def _reflect_and_improve(self, source_text: str, initial_translation: str, tgt_lang: str,
                             segment_av_ctx: Dict[str, Any], segment_idx: int) -> str:
        """
        结合片段级 AV 上下文优化翻译结果（反思机制）
        :param source_text: 源文本
        :param initial_translation: 基础翻译结果
        :param tgt_lang: 目标语言代码
        :param segment_av_ctx: 当前片段的 AV 上下文
        :param segment_idx: 片段索引（用于日志调试）
        :return: 优化后的翻译结果
        """
        # 提取 AV 上下文关键信息（处理空值）
        env = segment_av_ctx.get("environment", "未检测到").replace("检测到: ", "").strip()
        emotion = segment_av_ctx.get("emotion", "未检测到").replace("推断: ", "").strip()
        activity = segment_av_ctx.get("activity", "未检测到").replace("推断: ", "").strip()
        scene_desc = segment_av_ctx.get("description", "无场景描述").strip()
        timestamp = segment_av_ctx.get("timestamp", 0.0)

        # 构建反思提示词（明确要求结合场景优化）
        prompt = f"""你是专业的字幕翻译助手，需严格根据以下场景信息优化翻译结果：

=== 当前字幕片段信息 ===
片段索引：{segment_idx}
源文本：{source_text}
基础翻译：{initial_translation}
场景环境：{env}
人物情感：{emotion}
活动状态：{activity}
详细场景：{scene_desc}
片段时间戳：{timestamp:.2f}s

=== 翻译优化要求 ===
1. 准确性：必须严格匹配源文本含义，不增删语义；
2. 场景适配：根据当前场景调整语气（如室内交谈用正式语气，室外运动用活泼语气）、用词（如“拿着滑板”对应运动场景，“看着照片”对应静态场景）；
3. 简洁性：字幕需简短精炼，符合视听同步（避免过长句子）；
4. 目标语言：翻译成{self._get_lang_name(tgt_lang)}，确保语法正确、表达自然；
5. 输出格式：仅返回优化后的翻译文本，不要添加任何额外说明、标点或格式符号。

请直接输出优化后的翻译结果："""

        try:
            # 调用反思模型生成优化结果
            response = self.reflector(
                prompt,
                max_new_tokens=128,  # 限制优化结果长度
                do_sample=False,     # 关闭采样，确保结果稳定
                num_return_sequences=1,
                pad_token_id=self.reflector.tokenizer.eos_token_id,
                eos_token_id=self.reflector.tokenizer.eos_token_id
            )[0]["generated_text"]

            # 提取优化结果（去除提示词残留，仅保留翻译文本）
            optimized = response.split("请直接输出优化后的翻译结果：")[-1].strip()
            # 过滤空结果（若优化失败，返回基础翻译）
            if not optimized or len(optimized) < 1:
                logger.warning(f"Segment {segment_idx} reflection failed (empty result), using initial translation.")
                return initial_translation

            logger.debug(f"Segment {segment_idx} reflection success: '{initial_translation}' -> '{optimized}'")
            return optimized

        except Exception as e:
            logger.error(f"Segment {segment_idx} reflection failed: {e}", exc_info=True)
            # 反思失败时返回基础翻译，保证流程不中断
            return initial_translation

    def _calculate_batch_qe_scores(self, source_texts: List[str], translated_texts: List[str]) -> List[float]:
        """批量计算翻译质量（QE）分数（基于句子相似度）"""
        if not source_texts or not translated_texts or len(source_texts) != len(translated_texts):
            logger.warning("Invalid input for QE score calculation.")
            return [0.0]*len(source_texts)

        try:
            # 编码源文本和翻译文本（生成向量）
            src_embeddings = self.qe_model.encode(source_texts, convert_to_tensor=True, show_progress_bar=False)
            trans_embeddings = self.qe_model.encode(translated_texts, convert_to_tensor=True, show_progress_bar=False)

            # 计算余弦相似度（相似度越高，翻译质量越好）
            similarities = util.cos_sim(src_embeddings, trans_embeddings).diag().cpu().numpy()
            # 转换为 0-1 分数（余弦相似度本身范围为 [-1,1]，这里取绝对值后归一化）
            qe_scores = [float(abs(sim)) for sim in similarities]

            logger.info(f"Batch QE score calculation completed: avg_score={round(sum(qe_scores)/len(qe_scores), 2)}")
            return qe_scores

        except Exception as e:
            logger.error(f"QE score calculation failed: {e}", exc_info=True)
            return [0.0]*len(source_texts)

    def _get_lang_name(self, lang_code: str) -> str:
        """将语言代码转换为中文名称（用于提示词）"""
        lang_names = {
            'zh': '中文', 'zh-cn': '中文', 'en': '英文', 'ja': '日文', 'ko': '韩文',
            'fr': '法文', 'de': '德文', 'es': '西班牙文', 'ru': '俄文', 'ar': '阿拉伯文',
            'pt': '葡萄牙文', 'it': '意大利文', 'nl': '荷兰文', 'pl': '波兰文'
        }
        return lang_names.get(lang_code.lower(), lang_code)

    def _cleanup_vram(self):
        """清理模型占用的显存/内存"""
        if self.nmt_model:
            del self.nmt_model
        if self.reflector:
            del self.reflector
        if self.qe_model:
            del self.qe_model

        # 清理 PyTorch 缓存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✅ VRAM/CPU memory cleaned up successfully.")

    def get_supported_languages(self) -> Dict[str, List[str]]:
        """返回支持的语言列表（用于前端展示）"""
        return {
            "whisper": ["auto", "en", "zh", "ja", "ko", "fr", "de", "es", "ru", "ar", "hi", "pt", "it", "nl", "pl"],
            "nmt": ["en", "zh", "zh-cn", "ja", "ko", "fr", "de", "es", "ru", "ar", "hi", "pt", "it", "nl", "pl"]
        }

    def __del__(self):
        """对象销毁时自动清理资源"""
        self._cleanup_vram()