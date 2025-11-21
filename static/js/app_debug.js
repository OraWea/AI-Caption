// AI字幕翻译应用主JavaScript文件 - 增强调试版本

let currentFile = null;
let transcriptionResult = null;
let translationResult = null;
let currentStep = 1;
let currentTheme = 'light';

// 添加详细的错误日志
function logError(functionName, error, details = {}) {
    console.error(`[${functionName}] 错误:`, error);
    console.error(`[${functionName}] 详细信息:`, details);

    // 发送错误到后端（可选）
    fetch('/api/log-error', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            function: functionName,
            error: error.message || error,
            details: details,
            timestamp: new Date().toISOString()
        })
    }).catch(err => console.error('日志发送失败:', err));
}

// 初始化
document.addEventListener('DOMContentLoaded', function () {
    console.log('应用初始化开始...');
    initializeEventListeners();
    loadTheme();
    loadLanguages().then(() => {
        console.log('语言列表加载完成');
    }).catch(err => {
        logError('loadLanguages', err);
    });
});

// 初始化事件监听器
function initializeEventListeners() {
    console.log('初始化事件监听器...');

    // 文件上传区域
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');

    if (uploadArea) {
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
    } else {
        logError('initializeEventListeners', new Error('上传区域元素未找到'));
    }

    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    } else {
        logError('initializeEventListeners', new Error('文件输入元素未找到'));
    }

    // 按钮事件
    const startTranscriptionBtn = document.getElementById('start-transcription');
    const startTranslationBtn = document.getElementById('start-translation');
    const downloadSubtitleBtn = document.getElementById('download-subtitle');
    const processAnotherBtn = document.getElementById('process-another');

    if (startTranscriptionBtn) {
        startTranscriptionBtn.addEventListener('click', startTranscription);
    } else {
        logError('initializeEventListeners', new Error('开始转录按钮未找到'));
    }

    if (startTranslationBtn) {
        startTranslationBtn.addEventListener('click', startTranslation);
    } else {
        logError('initializeEventListeners', new Error('开始翻译按钮未找到'));
    }

    if (downloadSubtitleBtn) {
        downloadSubtitleBtn.addEventListener('click', downloadSubtitle);
    } else {
        logError('initializeEventListeners', new Error('下载字幕按钮未找到'));
    }

    if (processAnotherBtn) {
        processAnotherBtn.addEventListener('click', resetProcess);
    } else {
        logError('initializeEventListeners', new Error('处理另一个文件按钮未找到'));
    }

    // 主题切换按钮
    const themeToggleBtn = document.getElementById('theme-toggle');
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', toggleTheme);
        console.log('主题切换按钮已绑定');
    } else {
        logError('initializeEventListeners', new Error('主题切换按钮未找到'));
    }

    console.log('事件监听器初始化完成');
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
        console.log(`接收到拖拽文件: ${files[0].name}`);
        handleFile(files[0]);
    }
}

// 文件选择处理
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        console.log(`选择文件: ${file.name}, 大小: ${file.size} 字节`);
        handleFile(file);
    }
}

// 处理文件
function handleFile(file) {
    console.log('处理文件:', file.name);

    // 验证文件类型
    const supportedTypes = [
        'audio/mp3', 'audio/wav', 'audio/m4a', 'audio/flac', 'audio/aac',
        'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo', 'video/webm'
    ];

    console.log(`文件类型: ${file.type}`);

    if (!supportedTypes.some(type => file.type.includes(type.split('/')[1]))) {
        const errorMsg = `不支持的文件格式: ${file.type}`;
        console.error(errorMsg);
        showAlert(errorMsg, 'danger');
        return;
    }

    // 验证文件大小 (500MB)
    if (file.size > 500 * 1024 * 1024) {
        const errorMsg = '文件过大，最大支持500MB';
        console.error(errorMsg);
        showAlert(errorMsg, 'danger');
        return;
    }

    currentFile = file;

    // 更新UI
    const selectedFileElement = document.getElementById('selected-file');
    if (selectedFileElement) {
        selectedFileElement.textContent = file.name;
    }

    const fileInfoElement = document.getElementById('file-info');
    if (fileInfoElement) {
        fileInfoElement.style.display = 'block';
    }

    console.log('文件处理完成，准备进入转录步骤');

    // 自动进入下一步
    setTimeout(() => goToStep(2), 1000);
}

// 加载语言列表
async function loadLanguages() {
    console.log('开始加载语言列表...');

    try {
        const response = await fetch('/api/languages');
        console.log(`语言列表API响应状态: ${response.status}`);

        if (!response.ok) {
            throw new Error(`语言列表API失败: ${response.status}`);
        }

        const data = await response.json();
        console.log('语言数据加载成功:', Object.keys(data).length, '个类别');

        // 填充源语言选择器
        const sourceLangSelect = document.getElementById('source-language');
        if (sourceLangSelect && data.whisper_languages) {
            Object.entries(data.whisper_languages).forEach(([code, name]) => {
                const option = document.createElement('option');
                option.value = code;
                option.textContent = `${name} (${code})`;
                sourceLangSelect.appendChild(option);
            });
            console.log('源语言选择器填充完成');
        } else {
            logError('loadLanguages', new Error('源语言选择器或数据不存在'));
        }

        // 填充目标语言选择器
        const targetLangSelect = document.getElementById('target-language');
        if (targetLangSelect && data.translator_languages) {
            Object.entries(data.translator_languages).forEach(([code, name]) => {
                if (code !== 'auto') {
                    const option = document.createElement('option');
                    option.value = code;
                    option.textContent = name;
                    targetLangSelect.appendChild(option);
                }
            });
            console.log('目标语言选择器填充完成');
        } else {
            logError('loadLanguages', new Error('目标语言选择器或数据不存在'));
        }

    } catch (error) {
        logError('loadLanguages', error);
        showAlert('加载语言列表失败', 'danger');
    }
}

// 开始转录
async function startTranscription() {
    console.log('开始转录流程...');

    if (!currentFile) {
        console.error('没有当前文件');
        showAlert('请先上传文件', 'warning');
        return;
    }

    console.log(`转录文件: ${currentFile.name}`);

    const formData = new FormData();
    formData.append('file', currentFile);

    // 提前定义变量，避免作用域问题
    const progressElement = document.getElementById('transcription-progress');
    const startButton = document.getElementById('start-transcription');

    try {
        // 显示进度
        if (progressElement) {
            progressElement.style.display = 'block';
        }
        if (startButton) {
            startButton.disabled = true;
        }

        console.log('开始上传文件...');

        // 上传文件
        const uploadResponse = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        console.log(`文件上传响应状态: ${uploadResponse.status}`);

        if (!uploadResponse.ok) {
            const errorText = await uploadResponse.text();
            throw new Error(`文件上传失败: ${uploadResponse.status} - ${errorText}`);
        }

        const uploadData = await uploadResponse.json();
        console.log('文件上传成功:', uploadData);

        // 音频转录
        const sourceLang = document.getElementById('source-language').value;
        console.log(`源语言: ${sourceLang}`);

        console.log('开始音频转录...');
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

        console.log(`音频转录响应状态: ${transcribeResponse.status}`);

        if (!transcribeResponse.ok) {
            const errorText = await transcribeResponse.text();
            throw new Error(`音频转录失败: ${transcribeResponse.status} - ${errorText}`);
        }

        transcriptionResult = await transcribeResponse.json();
        console.log('音频转录成功:', {
            language: transcriptionResult.language,
            textLength: transcriptionResult.text?.length,
            segmentsCount: transcriptionResult.segments?.length
        });

        // 检查转录结果是否有效
        if (!transcriptionResult.segments || transcriptionResult.segments.length === 0) {
            console.warn('警告: 转录结果为空或没有段落');
            showAlert('警告: 没有检测到语音内容，请检查音频文件', 'warning');
        }

        // 进入下一步
        goToStep(3);

    } catch (error) {
        logError('startTranscription', error);
        showAlert('转录失败: ' + error.message, 'danger');

        // 恢复按钮状态
        if (progressElement) progressElement.style.display = 'none';
        if (startButton) startButton.disabled = false;
    }
}

// 开始翻译
async function startTranslation() {
    console.log('开始翻译流程...');

    if (!transcriptionResult) {
        console.error('没有转录结果');
        showAlert('请先完成音频转录', 'warning');
        return;
    }

    console.log('转录结果:', {
        language: transcriptionResult.language,
        segmentsCount: transcriptionResult.segments?.length,
        textLength: transcriptionResult.text?.length
    });

    const skipTranslation = document.getElementById('skip-translation').checked;

    if (skipTranslation) {
        console.log('用户选择跳过翻译');
        translationResult = {
            segments: transcriptionResult.segments
        };
        showSubtitlePreview();
        goToStep(4);
        return;
    }

    // 检查是否有有效的段落可以翻译
    if (!transcriptionResult.segments || transcriptionResult.segments.length === 0) {
        console.warn('没有有效的字幕段落可以翻译');
        showAlert('警告: 没有可翻译的字幕内容', 'warning');
        translationResult = {
            segments: []
        };
        showSubtitlePreview();
        goToStep(4);
        return;
    }

    try {
        // 显示进度
        const progressElement = document.getElementById('translation-progress');
        const startButton = document.getElementById('start-translation');

        if (progressElement) {
            progressElement.style.display = 'block';
        }
        if (startButton) {
            startButton.disabled = true;
        }

        const targetLang = document.getElementById('target-language').value;
        const sourceLang = document.getElementById('source-language').value;

        console.log(`翻译参数 - 源语言: ${sourceLang}, 目标语言: ${targetLang}`);
        console.log(`需要翻译的段落数: ${transcriptionResult.segments.length}`);

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

        console.log(`翻译响应状态: ${response.status}`);

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`字幕翻译失败: ${response.status} - ${errorText}`);
        }

        translationResult = await response.json();
        console.log('翻译成功:', {
            segmentsCount: translationResult.segments?.length
        });

        // 显示预览
        showSubtitlePreview();

        // 进入下一步
        goToStep(4);

    } catch (error) {
        logError('startTranslation', error);
        showAlert('翻译失败: ' + error.message, 'danger');

        // 恢复按钮状态
        if (progressElement) progressElement.style.display = 'none';
        if (startButton) startButton.disabled = false;
    }
}

// 显示字幕预览
function showSubtitlePreview() {
    console.log('显示字幕预览...');

    if (!translationResult || !translationResult.segments) {
        console.warn('没有翻译结果或段落');
        return;
    }

    const preview = document.getElementById('subtitle-preview');
    if (!preview) {
        logError('showSubtitlePreview', new Error('字幕预览元素未找到'));
        return;
    }

    console.log(`预览段落数: ${translationResult.segments.length}`);

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
    console.log('字幕预览显示完成');
}

// 下载字幕
async function downloadSubtitle() {
    console.log('开始下载字幕...');

    if (!translationResult || !translationResult.segments) {
        console.error('没有可下载的字幕');
        showAlert('没有可下载的字幕', 'warning');
        return;
    }

    console.log(`下载字幕参数 - 段落数: ${translationResult.segments.length}`);

    try {
        const format = document.getElementById('subtitle-format').value;
        const filename = currentFile ? currentFile.name : 'subtitle';

        console.log(`下载格式: ${format}, 文件名: ${filename}`);

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

        console.log(`字幕生成响应状态: ${response.status}`);

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`字幕文件生成失败: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        console.log('字幕生成成功:', data);

        // 下载文件
        console.log(`开始下载: ${data.download_url}`);
        window.location.href = data.download_url;

        // 清理服务器文件
        if (currentFile) {
            try {
                await fetch('/api/process-complete', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        file_path: currentFile.path
                    })
                });
                console.log('服务器文件清理完成');
            } catch (cleanupError) {
                console.warn('服务器文件清理失败:', cleanupError);
            }
        }

        showAlert('字幕下载成功！', 'success');

    } catch (error) {
        logError('downloadSubtitle', error);
        showAlert('下载失败: ' + error.message, 'danger');
    }
}

// 重置处理流程
function resetProcess() {
    console.log('重置处理流程...');

    currentFile = null;
    transcriptionResult = null;
    translationResult = null;
    currentStep = 1;

    // 重置UI
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const transcriptionProgress = document.getElementById('transcription-progress');
    const translationProgress = document.getElementById('translation-progress');
    const startTranscription = document.getElementById('start-transcription');
    const startTranslation = document.getElementById('start-translation');
    const subtitlePreview = document.getElementById('subtitle-preview');

    if (fileInput) fileInput.value = '';
    if (fileInfo) fileInfo.style.display = 'none';
    if (transcriptionProgress) transcriptionProgress.style.display = 'none';
    if (translationProgress) translationProgress.style.display = 'none';
    if (startTranscription) startTranscription.disabled = false;
    if (startTranslation) startTranslation.disabled = false;
    if (subtitlePreview) subtitlePreview.innerHTML = '';

    goToStep(1);
    console.log('处理流程重置完成');
}

// 步骤导航
function goToStep(step) {
    console.log(`切换到步骤 ${step}`);

    // 隐藏所有步骤
    document.querySelectorAll('.progress-step').forEach(el => {
        el.classList.remove('active');
    });

    // 显示当前步骤
    const stepIds = ['upload-step', 'transcribe-step', 'translate-step', 'download-step'];
    const stepElement = document.getElementById(stepIds[step - 1]);

    if (stepElement) {
        stepElement.classList.add('active');
        console.log(`步骤 ${step} 已激活`);
    } else {
        logError('goToStep', new Error(`步骤元素未找到: ${stepIds[step - 1]}`));
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
    console.log(`显示提示: [${type}] ${message}`);

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

// 全局错误处理
window.addEventListener('error', function (event) {
    logError('GlobalError', event.error, {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        message: event.message
    });
});

window.addEventListener('unhandledrejection', function (event) {
    logError('UnhandledRejection', event.reason);
});

// 主题切换功能
function toggleTheme() {
    console.log('切换主题...');
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    applyTheme(currentTheme);
    localStorage.setItem('theme', currentTheme);
    console.log(`当前主题: ${currentTheme}`);
}

function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    console.log(`加载保存的主题: ${savedTheme}`);
    currentTheme = savedTheme;
    applyTheme(currentTheme);
}

function applyTheme(theme) {
    console.log(`应用主题: ${theme}`);

    const themeIcon = document.getElementById('theme-icon');
    const themeText = document.getElementById('theme-text');

    if (!themeIcon || !themeText) {
        console.error('主题切换元素未找到');
        return;
    }

    if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
        themeIcon.className = 'fas fa-sun';
        themeText.textContent = '日间模式';
        console.log('已切换到黑夜模式');
    } else {
        document.documentElement.removeAttribute('data-theme');
        themeIcon.className = 'fas fa-moon';
        themeText.textContent = '夜间模式';
        console.log('已切换到白天模式');
    }
}

console.log('增强版JavaScript应用加载完成');