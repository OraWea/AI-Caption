import srt
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import timedelta
import json

logger = logging.getLogger(__name__)

class SubtitleGenerator:
    def __init__(self):
        """初始化字幕生成器"""
        pass
    
    def create_srt_subtitle(self, segments: List[Dict[str, Any]], output_path: str) -> str:
        """
        创建SRT格式字幕文件
        
        Args:
            segments: 字幕段落列表
            output_path: 输出文件路径
            
        Returns:
            输出文件路径
        """
        try:
            logger.info(f"正在创建SRT字幕文件: {output_path}")
            
            subtitles = []
            for i, segment in enumerate(segments, 1):
                start_time = timedelta(seconds=segment["start"])
                end_time = timedelta(seconds=segment["end"])
                text = segment["text"].strip()
                
                subtitle = srt.Subtitle(
                    index=i,
                    start=start_time,
                    end=end_time,
                    content=text
                )
                subtitles.append(subtitle)
            
            # 生成SRT内容
            srt_content = srt.compose(subtitles)
            
            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            logger.info(f"SRT字幕文件创建成功: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"SRT字幕文件创建失败: {e}")
            raise Exception(f"SRT字幕文件创建失败: {e}")
    
    def create_vtt_subtitle(self, segments: List[Dict[str, Any]], output_path: str) -> str:
        """
        创建WebVTT格式字幕文件
        
        Args:
            segments: 字幕段落列表
            output_path: 输出文件路径
            
        Returns:
            输出文件路径
        """
        try:
            logger.info(f"正在创建WebVTT字幕文件: {output_path}")
            
            vtt_content = "WEBVTT\n\n"
            
            for segment in segments:
                start_time = self._seconds_to_vtt_time(segment["start"])
                end_time = self._seconds_to_vtt_time(segment["end"])
                text = segment["text"].strip()
                
                vtt_content += f"{start_time} --> {end_time}\n"
                vtt_content += f"{text}\n\n"
            
            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(vtt_content)
            
            logger.info(f"WebVTT字幕文件创建成功: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"WebVTT字幕文件创建失败: {e}")
            raise Exception(f"WebVTT字幕文件创建失败: {e}")
    
    def create_json_subtitle(self, segments: List[Dict[str, Any]], output_path: str) -> str:
        """
        创建JSON格式字幕文件
        
        Args:
            segments: 字幕段落列表
            output_path: 输出文件路径
            
        Returns:
            输出文件路径
        """
        try:
            logger.info(f"正在创建JSON字幕文件: {output_path}")
            
            subtitle_data = {
                "segments": segments,
                "count": len(segments)
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(subtitle_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"JSON字幕文件创建成功: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"JSON字幕文件创建失败: {e}")
            raise Exception(f"JSON字幕文件创建失败: {e}")
    
    def create_subtitle(self, segments: List[Dict[str, Any]], output_path: str, 
                       format: str = "srt") -> str:
        """
        创建字幕文件（通用接口）
        
        Args:
            segments: 字幕段落列表
            output_path: 输出文件路径
            format: 字幕格式 (srt, vtt, json)
            
        Returns:
            输出文件路径
        """
        format = format.lower()
        
        if format == "srt":
            return self.create_srt_subtitle(segments, output_path)
        elif format == "vtt":
            return self.create_vtt_subtitle(segments, output_path)
        elif format == "json":
            return self.create_json_subtitle(segments, output_path)
        else:
            raise ValueError(f"不支持的字幕格式: {format}")
    
    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """
        将秒数转换为WebVTT时间格式
        
        Args:
            seconds: 秒数
            
        Returns:
            WebVTT时间格式字符串
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        milliseconds = int((secs % 1) * 1000)
        secs = int(secs)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    
    def format_segments_for_display(self, segments: List[Dict[str, Any]]) -> str:
        """
        格式化字幕段落用于显示
        
        Args:
            segments: 字幕段落列表
            
        Returns:
            格式化后的字符串
        """
        formatted_text = ""
        
        for i, segment in enumerate(segments, 1):
            start_time = self._format_time(segment["start"])
            end_time = self._format_time(segment["end"])
            text = segment["text"].strip()
            
            formatted_text += f"{i}\n"
            formatted_text += f"{start_time} --> {end_time}\n"
            formatted_text += f"{text}\n\n"
        
        return formatted_text
    
    def _format_time(self, seconds: float) -> str:
        """
        格式化时间显示
        
        Args:
            seconds: 秒数
            
        Returns:
            格式化时间字符串
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"