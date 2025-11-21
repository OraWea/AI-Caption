import os
import uuid
import logging
from typing import Optional, Tuple
from werkzeug.utils import secure_filename
from config import Config

logger = logging.getLogger(__name__)

class FileHandler:
    def __init__(self):
        """初始化文件处理器"""
        self.upload_folder = Config.UPLOAD_FOLDER
        self.allowed_extensions = Config.SUPPORTED_FORMATS
        
    def save_uploaded_file(self, file, subfolder: str = "") -> Tuple[str, str]:
        """
        保存上传的文件
        
        Args:
            file: Flask文件对象
            subfolder: 子文件夹
            
        Returns:
            文件路径和文件名
        """
        try:
            if not file or not file.filename:
                raise ValueError("没有上传文件")
            
            # 检查文件扩展名
            filename = secure_filename(file.filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext not in self.allowed_extensions:
                raise ValueError(f"不支持的文件格式: {file_ext}")
            
            # 生成唯一文件名
            unique_filename = f"{uuid.uuid4().hex}{file_ext}"
            
            # 创建保存路径
            if subfolder:
                save_dir = os.path.join(self.upload_folder, subfolder)
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, unique_filename)
            else:
                file_path = os.path.join(self.upload_folder, unique_filename)
            
            # 保存文件
            file.save(file_path)
            
            logger.info(f"文件保存成功: {file_path}")
            return file_path, unique_filename
            
        except Exception as e:
            logger.error(f"文件保存失败: {e}")
            raise Exception(f"文件保存失败: {e}")
    
    def delete_file(self, file_path: str) -> bool:
        """
        删除文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否删除成功
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"文件删除成功: {file_path}")
                return True
            else:
                logger.warning(f"文件不存在: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"文件删除失败: {e}")
            return False
    
    def get_file_size(self, file_path: str) -> int:
        """
        获取文件大小
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件大小（字节）
        """
        try:
            return os.path.getsize(file_path)
        except Exception as e:
            logger.error(f"获取文件大小失败: {e}")
            return 0
    
    def get_file_extension(self, filename: str) -> str:
        """
        获取文件扩展名
        
        Args:
            filename: 文件名
            
        Returns:
            扩展名（包含点）
        """
        return os.path.splitext(filename)[1].lower()
    
    def is_supported_format(self, filename: str) -> bool:
        """
        检查文件格式是否受支持
        
        Args:
            filename: 文件名
            
        Returns:
            是否受支持
        """
        file_ext = self.get_file_extension(filename)
        return file_ext in self.allowed_extensions
    
    def generate_output_filename(self, original_filename: str, suffix: str = "", 
                               extension: str = ".srt") -> str:
        """
        生成输出文件名
        
        Args:
            original_filename: 原始文件名
            suffix: 后缀
            extension: 扩展名
            
        Returns:
            输出文件名
        """
        base_name = os.path.splitext(original_filename)[0]
        if suffix:
            return f"{base_name}_{suffix}{extension}"
        else:
            return f"{base_name}{extension}"
    
    def create_temp_directory(self, prefix: str = "temp_") -> str:
        """
        创建临时目录
        
        Args:
            prefix: 目录前缀
            
        Returns:
            临时目录路径
        """
        temp_dir = os.path.join(Config.TEMP_FOLDER, f"{prefix}{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir
    
    def cleanup_temp_files(self, temp_dir: str) -> bool:
        """
        清理临时文件
        
        Args:
            temp_dir: 临时目录路径
            
        Returns:
            是否清理成功
        """
        try:
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"临时目录清理成功: {temp_dir}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"清理临时目录失败: {e}")
            return False