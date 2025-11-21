#!/bin/bash

# AI字幕生成翻译系统 - 安装脚本

echo "🚀 AI字幕生成翻译系统安装脚本"
echo "=================================="

# 检查Python版本
echo "检查Python版本..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
if [[ -z "$python_version" ]]; then
    echo "❌ Python3未安装，请先安装Python3.7+"
    exit 1
fi

major_version=$(echo $python_version | cut -d. -f1)
minor_version=$(echo $python_version | cut -d. -f2)

if [[ $major_version -lt 3 ]] || [[ $major_version -eq 3 && $minor_version -lt 7 ]]; then
    echo "❌ Python版本过低，需要Python3.7+，当前版本: $python_version"
    exit 1
fi

echo "✅ Python版本检查通过: $python_version"

# 检查FFmpeg
echo "检查FFmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg未安装，某些功能可能无法使用"
    echo "建议安装FFmpeg: https://ffmpeg.org/download.html"
else
    echo "✅ FFmpeg已安装"
fi

# 创建虚拟环境（推荐）
echo "创建Python虚拟环境..."
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
    echo "✅ 虚拟环境创建完成"
else
    echo "虚拟环境已存在"
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo "安装Python依赖..."
pip install -r requirements.txt

if [[ $? -eq 0 ]]; then
    echo "✅ 依赖安装成功"
else
    echo "❌ 依赖安装失败"
    exit 1
fi

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p static/uploads
mkdir -p static/css
mkdir -p static/js
mkdir -p output
mkdir -p temp

# 检查模型下载
echo "检查Whisper模型..."
python3 -c "
import whisper
print('正在下载Whisper基础模型...')
model = whisper.load_model('base')
print('✅ Whisper模型加载成功')
" 2>/dev/null

if [[ $? -eq 0 ]]; then
    echo "✅ Whisper模型检查完成"
else
    echo "⚠️  Whisper模型检查失败，首次使用时会自动下载"
fi

# 创建启动脚本
cat > start.sh << 'EOF'
#!/bin/bash
# AI字幕生成翻译系统启动脚本

echo "启动AI字幕生成翻译系统..."

# 激活虚拟环境
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# 启动应用
echo "访问 http://localhost:5000 使用系统"
python app.py
EOF

chmod +x start.sh

# 创建Windows启动脚本
cat > start.bat << 'EOF'
@echo off
REM AI字幕生成翻译系统启动脚本

echo 启动AI字幕生成翻译系统...

REM 激活虚拟环境
call venv\Scripts\activate

REM 启动应用
echo 访问 http://localhost:5000 使用系统
python app.py
pause
EOF

echo ""
echo "🎉 安装完成！"
echo "==============="
echo "启动方式："
echo "  Linux/Mac: ./start.sh"
echo "  Windows: 双击 start.bat"
echo ""
echo "访问地址: http://localhost:5000"
echo ""
echo "命令行使用:"
echo "  python cli.py input.mp4 -t zh-cn"
echo ""
echo "首次使用会下载AI模型，可能需要一些时间"
echo "请确保网络连接正常"