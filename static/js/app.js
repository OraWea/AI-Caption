// AI字幕翻译应用主JavaScript文件

let currentFile = null;
let transcriptionResult = null;
let translationResult = null;
let currentStep = 1;
let currentTheme = 'light';

// 初始化
document.addEventListener('DOMContentLoaded', function () {
    initializeEventListeners();
    loadLanguages();
    loadTheme();
});

// 初始化事件监听器
function initializeEventListeners() {
    // 文件上传区域
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');

    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);

    // 按钮事件
    document.getElementById('start-transcription').addEventListener('click', startTranscription);
    document.getElementById('start-translation').addEventListener('click', startTranslation);
    document.getElementById('download-subtitle').addEventListener('click', downloadSubtitle);
    document.getElementById('process-another').addEventListener('click', resetProcess);

    // 主题切换
    document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
}

// 拖拽处理
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// 文件选择处理
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// 处理文件
function handleFile(file) {
    // 验证文件类型
    const supportedTypes = [
        'audio/mp3', 'audio/wav', 'audio/m4a', 'audio/flac', 'audio/aac',
        'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo', 'video/webm'
    ];

    if (!supportedTypes.some(type => file.type.includes(type.split('/')[1]))) {
        showAlert('不支持的文件格式', 'danger');
        return;
    }

    // 验证文件大小 (500MB)
    if (file.size > 500 * 1024 * 1024) {
        showAlert('文件过大，最大支持500MB', 'danger');
        return;
    }

    currentFile = file;
    document.getElementById('selected-file').textContent = file.name;
    document.getElementById('file-info').style.display = 'block';

    // 自动进入下一步
    setTimeout(() => goToStep(2), 1000);
}

// 加载语言列表
async function loadLanguages() {
    try {
        const response = await fetch('/api/languages');
        const data = await response.json();

        // 填充源语言选择器
        const sourceLangSelect = document.getElementById('source-language');
        Object.entries(data.whisper_languages).forEach(([code, name]) => {
            const option = document.createElement('option');
            option.value = code;
            option.textContent = `${name} (${code})`;
            sourceLangSelect.appendChild(option);
        });

        // 填充目标语言选择器
        const targetLangSelect = document.getElementById('target-language');
        Object.entries(data.translator_languages).forEach(([code, name]) => {
            if (code !== 'auto') {
                const option = document.createElement('option');
                option.value = code;
                option.textContent = name;
                targetLangSelect.appendChild(option);
            }
        });

    } catch (error) {
        console.error('加载语言列表失败:', error);
        showAlert('加载语言列表失败', 'danger');
    }
}

// 开始转录
async function startTranscription() {
    if (!currentFile) {
        showAlert('请先上传文件', 'warning');
        return;
    }

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        // 显示进度
        document.getElementById('transcription-progress').style.display = 'block';
        document.getElementById('start-transcription').disabled = true;

        // 上传文件
        const uploadResponse = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) {
            throw new Error('文件上传失败');
        }

        const uploadData = await uploadResponse.json();

        // 音频转录
        const sourceLang = document.getElementById('source-language').value;
        const transcribeResponse = await fetch('/api/transcribe', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                file_path: uploadData.file_path,
                language: sourceLang
            })
        });

        if (!transcribeResponse.ok) {
            throw new Error('音频转录失败');
        }

        transcriptionResult = await transcribeResponse.json();

        // 进入下一步
        goToStep(3);

    } catch (error) {
        console.error('转录失败:', error);
        showAlert('转录失败: ' + error.message, 'danger');

        // 恢复按钮状态
        document.getElementById('transcription-progress').style.display = 'none';
        document.getElementById('start-transcription').disabled = false;
    }
}

// 开始翻译
async function startTranslation() {
    if (!transcriptionResult) {
        showAlert('请先完成音频转录', 'warning');
        return;
    }

    const skipTranslation = document.getElementById('skip-translation').checked;

    if (skipTranslation) {
        // 跳过翻译，直接使用转录结果
        translationResult = {
            segments: transcriptionResult.segments
        };
        showSubtitlePreview();
        goToStep(4);
        return;
    }

    try {
        // 显示进度
        document.getElementById('translation-progress').style.display = 'block';
        document.getElementById('start-translation').disabled = true;

        const targetLang = document.getElementById('target-language').value;
        const sourceLang = document.getElementById('source-language').value;

        const response = await fetch('/api/translate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                segments: transcriptionResult.segments,
                target_language: targetLang,
                source_language: sourceLang
            })
        });

        if (!response.ok) {
            throw new Error('字幕翻译失败');
        }

        translationResult = await response.json();

        // 显示预览
        showSubtitlePreview();

        // 进入下一步
        goToStep(4);

    } catch (error) {
        console.error('翻译失败:', error);
        showAlert('翻译失败: ' + error.message, 'danger');

        // 恢复按钮状态
        document.getElementById('translation-progress').style.display = 'none';
        document.getElementById('start-translation').disabled = false;
    }
}

// 显示字幕预览
function showSubtitlePreview() {
    if (!translationResult || !translationResult.segments) {
        return;
    }

    const preview = document.getElementById('subtitle-preview');
    let html = '';

    translationResult.segments.forEach((segment, index) => {
        const startTime = formatTime(segment.start);
        const endTime = formatTime(segment.end);
        const text = segment.text;

        html += `<div class="subtitle-item">
            <div class="subtitle-time">${index + 1}. ${startTime} --> ${endTime}</div>
            <div class="subtitle-text">${text}</div>
        </div>`;
    });

    preview.innerHTML = html;
}

// 下载字幕
async function downloadSubtitle() {
    if (!translationResult || !translationResult.segments) {
        showAlert('没有可下载的字幕', 'warning');
        return;
    }

    try {
        const format = document.getElementById('subtitle-format').value;
        const filename = currentFile ? currentFile.name : 'subtitle';

        const response = await fetch('/api/generate-subtitle', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                segments: translationResult.segments,
                format: format,
                filename: filename
            })
        });

        if (!response.ok) {
            throw new Error('字幕文件生成失败');
        }

        const data = await response.json();

        // 下载文件
        window.location.href = data.download_url;

        // 清理服务器文件
        if (currentFile) {
            await fetch('/api/process-complete', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    file_path: currentFile.path
                })
            });
        }

        showAlert('字幕下载成功！', 'success');

    } catch (error) {
        console.error('下载失败:', error);
        showAlert('下载失败: ' + error.message, 'danger');
    }
}

// 重置处理流程
function resetProcess() {
    currentFile = null;
    transcriptionResult = null;
    translationResult = null;
    currentStep = 1;

    // 重置UI
    document.getElementById('file-input').value = '';
    document.getElementById('file-info').style.display = 'none';
    document.getElementById('transcription-progress').style.display = 'none';
    document.getElementById('translation-progress').style.display = 'none';
    document.getElementById('start-transcription').disabled = false;
    document.getElementById('start-translation').disabled = false;
    document.getElementById('subtitle-preview').innerHTML = '';

    goToStep(1);
}

// 步骤导航
function goToStep(step) {
    // 隐藏所有步骤
    document.querySelectorAll('.progress-step').forEach(el => {
        el.classList.remove('active');
    });

    // 显示当前步骤
    const stepElement = document.getElementById(['upload-step', 'transcribe-step', 'translate-step', 'download-step'][step - 1]);
    if (stepElement) {
        stepElement.classList.add('active');
    }

    // 更新步骤指示器
    document.querySelectorAll('.step').forEach((el, index) => {
        el.classList.remove('active', 'completed');
        if (index < step - 1) {
            el.classList.add('completed');
        } else if (index === step - 1) {
            el.classList.add('active');
        }
    });

    currentStep = step;
}

// 格式化时间
function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 1000);

    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
}

// 显示提示信息
function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.style.cssText = 'position: fixed; top: 80px; right: 20px; z-index: 9999; min-width: 300px; box-shadow: 0 10px 40px rgba(0,0,0,0.2);';
    alertDiv.innerHTML = `
        <strong><i class="fas fa-${type === 'success' ? 'check-circle' : type === 'danger' ? 'exclamation-circle' : 'info-circle'}"></i> ${message}</strong>
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    document.body.appendChild(alertDiv);

    // 3秒后自动关闭
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.classList.remove('show');
            setTimeout(() => alertDiv.remove(), 150);
        }
    }, 3000);
}

// 主题切换功能
function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    applyTheme(currentTheme);
    localStorage.setItem('theme', currentTheme);
}

function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    currentTheme = savedTheme;
    applyTheme(currentTheme);
}

function applyTheme(theme) {
    const themeIcon = document.getElementById('theme-icon');
    const themeText = document.getElementById('theme-text');

    if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
        themeIcon.className = 'fas fa-sun';
        themeText.textContent = '日间模式';
    } else {
        document.documentElement.removeAttribute('data-theme');
        themeIcon.className = 'fas fa-moon';
        themeText.textContent = '夜间模式';
    }
}