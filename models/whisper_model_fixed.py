import whisper
import os
import logging
import torch
import numpy as np
import cv2
from typing import Optional, Dict, Any, List
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2Tokenizer

# 导入视频处理器
from utils.video_processor import video_processor  # 绝对导入，直接从根目录找 utils

logger = logging.getLogger(__name__)

class WhisperModel:
    def __init__(self, model_name: str = "base", device: str = "cpu"):
        self.model_name = model_name
        self.whisper_device = device
        self.model = None
        self.vit_gpt2_model = None
        self.vit_processor = None
        self.gpt2_tokenizer = None
        self.vlm_device = None

        self.load_model()
        self.load_vit_gpt2_model()

    def load_model(self):
        try:
            logger.info(f"Loading Whisper model: {self.model_name} to {self.whisper_device}")
            self.model = whisper.load_model(self.model_name, device=self.whisper_device)
            logger.info(f"Whisper model {self.model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise Exception(f"Whisper model load failed: {e}")

    def load_vit_gpt2_model(self):
        vlm_model_id = "nlpconnect/vit-gpt2-image-captioning"
        self.vlm_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing ViT-GPT2 model '{vlm_model_id}' on device: {self.vlm_device}")

        try:
            # 使用 ViTImageProcessor 替代 deprecated 的 ViTFeatureExtractor
            self.vit_processor = ViTImageProcessor.from_pretrained(vlm_model_id)
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(vlm_model_id)
            self.vit_gpt2_model = VisionEncoderDecoderModel.from_pretrained(
                vlm_model_id,
                dtype=torch.float16 if self.vlm_device == "cuda" else torch.float32  # 使用 dtype 替代 torch_dtype
            ).to(self.vlm_device)

            # 处理 pad token 问题
            if self.gpt2_tokenizer.pad_token is None:
                self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
                logger.warning("Set pad_token to eos_token for GPT2Tokenizer")

            logger.info("✅ ViT-GPT2 model and components loaded successfully.")
        except Exception as e:
            logger.error(f"ViT-GPT2 load failed: {e}")
            self.vit_gpt2_model = None

    def _get_segment_av_context(self, video_path: str, segment_start: float, segment_end: float) -> Dict[str, Any]:
        """为单个字幕片段生成对应的AV上下文"""
        # 计算片段中间时间戳（取中间点，避免场景过渡）
        mid_timestamp = (segment_start + segment_end) / 2

        # 提取该时间戳的视频帧
        frame = video_processor.extract_frame_at_time(video_path, mid_timestamp)
        if frame is None:
            logger.warning(f"片段 {segment_start:.2f}s-{segment_end:.2f}s 无法提取帧，使用默认上下文")
            return {
                "environment": "未检测到",
                "emotion": "未检测到",
                "activity": "未检测到",
                "description": "片段场景提取失败",
                "timestamp": round(mid_timestamp, 2)
            }

        # 用ViT-GPT2生成场景信息
        try:
            # 图像预处理
            pixel_values = self.vit_processor(
                images=frame,
                return_tensors="pt"
            ).pixel_values.to(self.vlm_device, dtype=torch.float16 if self.vlm_device == "cuda" else torch.float32)

            # 生成描述文本
            gen_ids = self.vit_gpt2_model.generate(
                pixel_values,
                max_length=50,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            raw_desc = self.gpt2_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

            # 结构化解析
            return {
                "environment": f"检测到: {self._parse_environment(raw_desc)}",
                "emotion": f"推断: {self._parse_emotion(raw_desc)}",
                "activity": f"推断: {self._parse_activity(raw_desc)}",
                "description": raw_desc,
                "timestamp": round(mid_timestamp, 2)
            }
        except Exception as e:
            logger.error(f"片段 {segment_start:.2f}s-{segment_end:.2f} 场景解析失败：{e}")
            return {
                "environment": "未检测到",
                "emotion": "未检测到",
                "activity": "未检测到",
                "description": f"场景解析错误：{str(e)[:30]}",
                "timestamp": round(mid_timestamp, 2)
            }

    def transcribe(self, media_path: str, language: str = "auto", task: str = "transcribe",
                   video_source_path: Optional[str] = None) -> Dict[str, Any]:
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"Media file not found: {media_path}")

        try:
            logger.info(f"Starting transcription for: {media_path}")
            audio = whisper.load_audio(media_path)
            duration = len(audio) / whisper.audio.SAMPLE_RATE
            # 视频路径优先用传入的 video_source_path，无则用 media_path（音频文件时无效）
            video_path = video_source_path if video_source_path and os.path.exists(video_source_path) else media_path

            # Whisper 转录配置
            options = {"task": task}
            if language != "auto":
                options["language"] = language

            logger.info("Executing Whisper transcription...")
            self.model.to(self.whisper_device)
            result = self.model.transcribe(media_path, **options)

            # 整理字幕片段（为每个片段添加 av_context）
            segments = []
            for seg in result.get("segments", []):
                segment_start = seg["start"]
                segment_end = seg["end"]
                # 为当前片段生成AV上下文（仅当视频路径有效时）
                if video_path and video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    segment_av_ctx = self._get_segment_av_context(video_path, segment_start, segment_end)
                else:
                    segment_av_ctx = {
                        "environment": "非视频文件", 
                        "emotion": "非视频文件", 
                        "activity": "非视频文件", 
                        "description": "非视频文件无场景信息",
                        "timestamp": 0.0
                    }

                segments.append({
                    "start": segment_start,
                    "end": segment_end,
                    "text": seg["text"].strip(),
                    "av_context": segment_av_ctx  # 每个片段绑定自己的场景信息
                })

            # 保留全局AV上下文（可选，用于前端展示整体场景）
            global_av_ctx = self._get_av_context(video_path) if video_path else {}

            return {
                "text": result["text"].strip(),
                "segments": segments,
                "language": result["language"],
                "duration": duration,
                "av_context": global_av_ctx  # 全局场景（可选）
            }
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise Exception(f"Transcription error: {e}")

    def _get_av_context(self, video_path: str) -> Dict[str, Any]:
        """获取全局AV上下文（视频整体场景）"""
        default_ctx = {
            "environment": "未检测到", "person_count": "未检测到",
            "main_speaker_location": "未检测到", "emotion": "未检测到",
            "activity": "未检测到", "description": "AV 上下文提取失败"
        }

        if not self.vit_gpt2_model:
            logger.warning("ViT-GPT2 not initialized, skipping global AV context.")
            return default_ctx

        logger.info(f"[AV-CTX] Getting global context for video: {video_path}")
        # 提取视频中间帧作为全局场景代表
        total_duration = self._get_video_duration(video_path)
        if total_duration <= 0:
            logger.warning("Cannot get video duration, using default frame.")
            frame = self._extract_video_frame(video_path)
        else:
            frame = video_processor.extract_frame_at_time(video_path, total_duration / 2)

        if frame is None:
            logger.warning("[AV-CTX] No global frame extracted.")
            default_ctx["description"] = "无法提取视频帧"
            return default_ctx

        try:
            # 图像预处理
            pixel_values = self.vit_processor(
                images=frame,
                return_tensors="pt"
            ).pixel_values.to(self.vlm_device, dtype=torch.float16 if self.vlm_device == "cuda" else torch.float32)

            # 生成描述文本
            gen_ids = self.vit_gpt2_model.generate(
                pixel_values,
                max_length=50,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            raw_desc = self.gpt2_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

            # 结构化解析
            return {
                "environment": f"检测到: {self._parse_environment(raw_desc)}",
                "person_count": "未检测到",
                "main_speaker_location": "未检测到",
                "emotion": f"推断: {self._parse_emotion(raw_desc)}",
                "activity": f"推断: {self._parse_activity(raw_desc)}",
                "description": raw_desc
            }
        except Exception as e:
            logger.error(f"[AV-CTX] Global context extraction failed: {e}")
            default_ctx["description"] = f"全局场景错误: {str(e)[:50]}"
            return default_ctx

    def _get_video_duration(self, video_path: str) -> float:
        """获取视频时长（秒）"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return -1
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            return frame_count / fps if fps > 0 else 0
        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")
            return -1

    def _extract_video_frame(self, video_path: str, frame_idx: int = 0) -> Optional[np.ndarray]:
        """提取视频指定索引的帧（备用方法）"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return None

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                logger.warning(f"Failed to read frame {frame_idx} from video")
                return None
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return None

    # 辅助方法：解析环境（室内/室外）
    def _parse_environment(self, desc: str) -> str:
        outdoor_keywords = ["outdoor", "outside", "park", "street", "beach", "forest", "yard", "garden", "field", "mountain"]
        indoor_keywords = ["indoor", "inside", "room", "house", "office", "studio", "kitchen", "living room", "bedroom", "hall"]

        desc_lower = desc.lower()
        if any(kw in desc_lower for kw in outdoor_keywords):
            return "室外"
        elif any(kw in desc_lower for kw in indoor_keywords):
            return "室内"
        else:
            # 基于常见场景推断（如“桌子前”默认室内）
            return "室内" if any(kw in desc_lower for kw in ["table", "desk", "chair", "wall", "window"]) else "未知"

    # 辅助方法：解析情感
    def _parse_emotion(self, desc: str) -> str:
        positive_keywords = ["smiling", "happy", "laughing", "excited", "joyful", "cheerful", "grinning", "delighted"]
        calm_keywords = ["calm", "relaxed", "quiet", "still", "peaceful", "serene", "composed"]
        negative_keywords = ["sad", "angry", "upset", "frowning", "frustrated", "crying", "mad"]

        desc_lower = desc.lower()
        if any(kw in desc_lower for kw in positive_keywords):
            return "开心/兴奋"
        elif any(kw in desc_lower for kw in calm_keywords):
            return "平静/放松"
        elif any(kw in desc_lower for kw in negative_keywords):
            return "悲伤/愤怒"
        else:
            return "中性"

    # 辅助方法：解析动作
    def _parse_activity(self, desc: str) -> str:
        talking_keywords = ["talking", "speaking", "discussing", "interview", "chatting", "conversing"]
        action_keywords = ["holding", "using", "playing", "running", "walking", "skateboarding", "dancing", "eating", "drinking", "writing", "reading"]
        static_keywords = ["standing", "sitting", "posing", "looking", "watching", "listening", "sleeping"]

        desc_lower = desc.lower()
        if any(kw in desc_lower for kw in talking_keywords):
            return "交谈/说话"
        elif any(kw in desc_lower for kw in action_keywords):
            return "进行动作（持物/运动等）"
        elif any(kw in desc_lower for kw in static_keywords):
            return "静止状态（站立/坐姿等）"
        else:
            return "未知活动"