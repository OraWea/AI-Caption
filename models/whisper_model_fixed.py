# 修复版Whisper模型 - 无需FFmpeg依赖

import whisper
import os
import logging
from typing import Optional, Dict, Any
import numpy as np
import wave

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
        转录音频文件 - 修复版，无需FFmpeg依赖
        
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
            
            # 直接使用Whisper加载音频文件（不依赖FFmpeg）
            audio = self._load_audio_directly(audio_path)
            
            # 设置转录选项
            options = {
                "task": task,
                "verbose": False
            }
            
            if language != "auto":
                options["language"] = language
            
            # 执行转录
            result = self.model.transcribe(audio, **options)
            
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
    
    def _load_audio_directly(self, audio_path: str):
        """
        直接加载音频文件，不依赖FFmpeg
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频数据
        """
        try:
            # 尝试使用Whisper内置的音频加载功能
            return whisper.load_audio(audio_path)
        except Exception as e:
            logger.warning(f"Whisper内置音频加载失败: {e}")
            
            # 回退方案：手动读取WAV文件
            try:
                return self._load_wav_manually(audio_path)
            except Exception as e2:
                logger.error(f"手动音频加载也失败: {e2}")
                raise Exception(f"无法加载音频文件，请确保文件格式正确: {e}")
    
    def _load_wav_manually(self, audio_path: str):
        """
        手动读取WAV文件
        
        Args:
            audio_path: WAV文件路径
            
        Returns:
            音频数据数组
        """
        import wave
        import numpy as np
        
        with wave.open(audio_path, 'rb') as wav_file:
            # 获取音频参数
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            logger.info(f"WAV文件信息: {channels}通道, {sample_width}字节, {frame_rate}Hz, {n_frames}帧")
            
            # 读取音频数据
            audio_data = wav_file.readframes(n_frames)
            
            # 转换为numpy数组
            if sample_width == 1:  # 8位音频
                audio_array = np.frombuffer(audio_data, dtype=np.uint8)
                audio_array = audio_array.astype(np.float32) / 128.0 - 1.0
            elif sample_width == 2:  # 16位音频
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0
            else:
                raise ValueError(f"不支持的采样宽度: {sample_width}")
            
            # 如果是立体声，转换为单声道
            if channels > 1:
                audio_array = audio_array.reshape(-1, channels)
                audio_array = audio_array.mean(axis=1)
            
            # 重采样到16kHz（Whisper要求）
            if frame_rate != 16000:
                logger.info(f"重采样从 {frame_rate}Hz 到 16000Hz")
                from scipy import signal
                audio_array = signal.resample(audio_array, int(len(audio_array) * 16000 / frame_rate))
            
            return audio_array
    
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
            audio = self._load_audio_directly(audio_path)
            
            # 检测语言
            language_probs = self.model.detect_language(audio)
            detected_language = max(language_probs, key=language_probs.get)
            
            logger.info(f"检测到音频语言: {detected_language}")
            return detected_language
            
        except Exception as e:
            logger.error(f"语言检测失败: {e}")
            return "unknown"