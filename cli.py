#!/usr/bin/env python3
"""
AI字幕生成翻译系统 - 命令行版本
用于测试和批量处理音频文件
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.whisper_model_fixed import WhisperModel
from models.translator import Translator
from utils.audio_processor import AudioProcessor
from utils.subtitle_generator import SubtitleGenerator
from utils.file_handler import FileHandler
from config import Config

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def process_single_file(file_path, args):
    """处理单个文件"""
    logger = logging.getLogger(__name__)
    
    try:
        # 初始化组件
        whisper_model = WhisperModel(
            model_name=args.model_size,
            device=args.device
        )
        translator = Translator(service=args.translator)
        audio_processor = AudioProcessor()
        subtitle_generator = SubtitleGenerator()
        file_handler = FileHandler()
        
        logger.info(f"正在处理文件: {file_path}")
        
        # 创建临时目录
        temp_dir = file_handler.create_temp_directory()
        
        try:
            # 处理音频文件
            processed_audio_path = audio_processor.process_audio_for_transcription(
                file_path, temp_dir
            )
            
            # 音频转录
            logger.info("开始音频转录...")
            transcribe_result = whisper_model.transcribe(
                processed_audio_path,
                language=args.source_lang
            )
            
            logger.info(f"转录完成，语言: {transcribe_result['language']}")
            logger.info(f"转录文本: {transcribe_result['text'][:100]}...")
            
            # 如果需要翻译
            if args.target_lang and args.target_lang != 'none':
                logger.info(f"开始翻译到 {args.target_lang}...")
                translated_segments = translator.translate_segments(
                    transcribe_result['segments'],
                    args.target_lang,
                    transcribe_result['language']
                )
                segments_to_use = translated_segments
            else:
                segments_to_use = transcribe_result['segments']
            
            # 生成字幕文件
            output_filename = f"{Path(file_path).stem}_translated.{args.format}"
            output_path = os.path.join(args.output_dir, output_filename)
            
            subtitle_path = subtitle_generator.create_subtitle(
                segments_to_use, output_path, args.format
            )
            
            logger.info(f"字幕文件生成成功: {subtitle_path}")
            
            # 保存转录结果
            if args.save_transcript:
                transcript_path = os.path.join(args.output_dir, f"{Path(file_path).stem}_transcript.txt")
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(transcribe_result['text'])
                logger.info(f"转录文本保存成功: {transcript_path}")
            
            return {
                'success': True,
                'subtitle_file': subtitle_path,
                'transcript': transcribe_result['text'],
                'language': transcribe_result['language'],
                'duration': transcribe_result['duration']
            }
            
        finally:
            # 清理临时文件
            file_handler.cleanup_temp_files(temp_dir)
            
    except Exception as e:
        logger.error(f"处理文件失败: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='AI字幕生成翻译系统 - 命令行版本')
    parser.add_argument('input', help='输入文件或目录路径')
    parser.add_argument('-o', '--output-dir', default='output', help='输出目录')
    parser.add_argument('-l', '--source-lang', default='auto', help='源语言 (auto表示自动检测)')
    parser.add_argument('-t', '--target-lang', default='zh-cn', help='目标语言 (none表示不翻译)')
    parser.add_argument('-f', '--format', default='srt', choices=['srt', 'vtt', 'json'], help='输出格式')
    parser.add_argument('-m', '--model-size', default='base', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper模型大小')
    parser.add_argument('-d', '--device', default='cpu', choices=['cpu', 'cuda'], help='运行设备')
    parser.add_argument('--translator', default='google', help='翻译服务')
    parser.add_argument('--save-transcript', action='store_true', help='保存转录文本')
    parser.add_argument('--recursive', action='store_true', help='递归处理子目录')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.mp4', '.mp3', '.wav', '.m4a', '.flac', '.aac'],
                       help='处理的文件扩展名')
    
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取要处理的文件列表
    input_path = Path(args.input)
    files_to_process = []
    
    if input_path.is_file():
        files_to_process.append(str(input_path))
    elif input_path.is_dir():
        pattern = '**/*' if args.recursive else '*'
        for ext in args.extensions:
            files_to_process.extend(input_path.glob(pattern + ext))
        files_to_process = [str(f) for f in files_to_process]
    else:
        logger.error(f"输入路径不存在: {input_path}")
        return 1
    
    if not files_to_process:
        logger.warning("没有找到要处理的文件")
        return 0
    
    logger.info(f"找到 {len(files_to_process)} 个文件需要处理")
    
    # 处理文件
    success_count = 0
    failed_count = 0
    
    for i, file_path in enumerate(files_to_process, 1):
        logger.info(f"\n处理进度: {i}/{len(files_to_process)}")
        
        result = process_single_file(file_path, args)
        
        if result['success']:
            success_count += 1
            logger.info(f"✓ 处理成功: {file_path}")
        else:
            failed_count += 1
            logger.error(f"✗ 处理失败: {file_path} - {result['error']}")
    
    # 输出统计信息
    logger.info(f"\n处理完成!")
    logger.info(f"成功: {success_count} 个文件")
    logger.info(f"失败: {failed_count} 个文件")
    logger.info(f"总计: {len(files_to_process)} 个文件")
    
    return 0 if failed_count == 0 else 1

if __name__ == '__main__':
    sys.exit(main())