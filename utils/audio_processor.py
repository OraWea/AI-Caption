import os
import logging
from pydub import AudioSegment
from typing import Optional
import subprocess
import shutil

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        """初始化音频处理器"""
        self.supported_formats = ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    def extract_audio_from_video(self, video_path: str, output_path: str) -> str:
        """
        从视频中提取音频
        
        Args:
            video_path: 视频文件路径
            output_path: 输出音频文件路径
            
        Returns:
            音频文件路径
        """
        try:
            logger.info(f"正在从视频中提取音频: {video_path}")
            
            # 首先检查FFmpeg是否可用
            if not self.check_ffmpeg():
                logger.warning("FFmpeg未安装，尝试使用pydub直接处理")
                try:
                    # 使用pydub直接加载视频文件（如果支持）
                    video = AudioSegment.from_file(video_path)
                    # 导出为WAV格式
                    video.export(output_path, format='wav', 
                               parameters=["-ar", "16000", "-ac", "1"])
                    logger.info(f"使用pydub提取音频完成: {output_path}")
                    return output_path
                except Exception as pydub_error:
                    logger.error(f"pydub处理失败: {pydub_error}")
                    raise Exception(f"FFmpeg未安装且pydub无法处理此视频格式。请安装FFmpeg或上传音频文件。")
            
            # 使用ffmpeg提取音频
            command = [
                'ffmpeg', '-i', video_path,
                '-vn',  # 无视频
                '-acodec', 'pcm_s16le',  # PCM编码
                '-ar', '16000',  # 16kHz采样率
                '-ac', '1',  # 单声道
                '-y',  # 覆盖输出文件
                output_path
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"FFmpeg错误: {result.stderr}")
            
            logger.info(f"音频提取完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"音频提取失败: {e}")
            raise Exception(f"音频提取失败: {e}")
    
    def convert_audio_format(self, input_path: str, output_path: str, 
                           format: str = 'wav', sample_rate: int = 16000) -> str:
        """
        转换音频格式
        
        Args:
            input_path: 输入音频文件路径
            output_path: 输出音频文件路径
            format: 输出格式
            sample_rate: 采样率
            
        Returns:
            输出文件路径
        """
        try:
            logger.info(f"正在转换音频格式: {input_path} -> {output_path}")
            
            # 加载音频文件
            audio = AudioSegment.from_file(input_path)
            
            # 设置参数
            audio = audio.set_frame_rate(sample_rate)
            audio = audio.set_channels(1)  # 单声道
            
            # 导出音频
            audio.export(output_path, format=format)
            
            logger.info(f"音频格式转换完成: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"音频格式转换失败: {e}")
            raise Exception(f"音频格式转换失败: {e}")
    
    def process_audio_for_transcription(self, input_path: str, temp_dir: str) -> str:
        """
        处理音频文件以供转录使用
        
        Args:
            input_path: 输入文件路径
            temp_dir: 临时目录
            
        Returns:
            处理后的音频文件路径
        """
        try:
            file_ext = os.path.splitext(input_path)[1].lower()
            
            # 如果是视频文件，先提取音频
            if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                logger.info(f"检测到视频文件: {input_path}")
                audio_path = os.path.join(temp_dir, 'extracted_audio.wav')
                
                # 尝试提取音频，如果失败则提供有用的错误信息
                try:
                    audio_path = self.extract_audio_from_video(input_path, audio_path)
                except Exception as video_error:
                    logger.error(f"视频处理失败: {video_error}")
                    # 如果是FFmpeg相关问题，提供替代方案
                    if "FFmpeg" in str(video_error):
                        raise Exception(
                            "视频处理需要FFmpeg。请：\n"
                            "1. 安装FFmpeg: https://ffmpeg.org/download.html\n"
                            "2. 或将视频转换为音频文件后上传\n"
                            "3. 或直接使用音频文件（.wav, .mp3等）"
                        )
                    else:
                        raise video_error
            else:
                audio_path = input_path
            
            # 转换为适合转录的格式
            processed_path = os.path.join(temp_dir, 'processed_audio.wav')
            processed_path = self.convert_audio_format(audio_path, processed_path)
            
            return processed_path
            
        except Exception as e:
            logger.error(f"音频处理失败: {e}")
            raise Exception(f"音频处理失败: {e}")
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        获取音频时长
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            时长（秒）
        """
        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # 转换为秒
        except Exception as e:
            logger.error(f"获取音频时长失败: {e}")
            return 0.0
    
    def validate_audio_file(self, file_path: str) -> bool:
        """
        验证音频文件是否有效
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否有效
        """
        try:
            if not os.path.exists(file_path):
                return False
            
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_formats:
                return False
            
            # 尝试加载音频文件
            audio = AudioSegment.from_file(file_path)
            return len(audio) > 0
            
        except Exception as e:
            logger.error(f"音频文件验证失败: {e}")
            return False
    
    def check_ffmpeg(self) -> bool:
        """检查FFmpeg是否可用"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False