import whisper
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class WhisperModel:
    def __init__(self, model_name: str = "base", device: str = "cpu"):
        """
        初始化Whisper模型
        
        Args:
            model_name: 模型名称 (tiny, base, small, medium, large)
            device: 运行设备 (cpu, cuda)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.load_model()
    
    def load_model(self):
        """加载Whisper模型"""
        try:
            logger.info(f"正在加载Whisper模型: {self.model_name}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info(f"Whisper模型 {self.model_name} 加载成功")
        except Exception as e:
            logger.error(f"加载Whisper模型失败: {e}")
            raise Exception(f"无法加载Whisper模型: {e}")
    
    def transcribe(self, audio_path: str, language: str = "auto", 
                   task: str = "transcribe") -> Dict[str, Any]:
        """
        转录音频文件
        
        Args:
            audio_path: 音频文件路径
            language: 音频语言 (auto表示自动检测)
            task: 任务类型 (transcribe, translate)
            
        Returns:
            包含转录结果的字典
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        try:
            logger.info(f"开始转录音频: {audio_path}")
            
            # 设置转录选项
            options = {
                "task": task,
                "verbose": False
            }
            
            if language != "auto":
                options["language"] = language
            
            # 执行转录
            result = self.model.transcribe(audio_path, **options)
            
            logger.info(f"音频转录完成，耗时: {result.get('duration', 0):.2f}秒")
            
            return {
                "text": result["text"],
                "segments": result["segments"],
                "language": result.get("language", "unknown"),
                "duration": result.get("duration", 0)
            }
            
        except Exception as e:
            logger.error(f"音频转录失败: {e}")
            raise Exception(f"音频转录失败: {e}")
    
    def get_supported_languages(self) -> Dict[str, str]:
        """获取支持的语言列表"""
        return whisper.tokenizer.LANGUAGES
    
    def detect_language(self, audio_path: str) -> str:
        """
        检测音频语言
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            语言代码
        """
        try:
            # 加载音频
            audio = whisper.load_audio(audio_path)
            
            # 检测语言
            language_probs = self.model.detect_language(audio)
            detected_language = max(language_probs, key=language_probs.get)
            
            logger.info(f"检测到音频语言: {detected_language}")
            return detected_language
            
        except Exception as e:
            logger.error(f"语言检测失败: {e}")
            return "unknown"