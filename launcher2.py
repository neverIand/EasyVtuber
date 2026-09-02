import ctypes
import importlib.util
import os
import subprocess
import threading
import time
from collections import deque

import wx
import sys

from src.utils.student_models import scan_student_models
from src.utils.gpu_detect import has_nvidia_gpu
from src.utils.dml_devices import launcher_directml_choices
from src.utils.launcher_config import (
    DEFAULT_LAUNCHER_CONFIG,
    SAFETY_PRESETS,
    beta_mapper,
    build_launch_command,
    describe_ram_cache,
    infer_safety_preset,
    load_launcher_config,
    min_cutoff_mapper,
    save_launcher_config,
)
from src.utils.preview_ipc import (
    PREVIEW_FPS,
    PreviewSharedBuffer,
)

ctypes.windll.shcore.SetProcessDpiAwareness(1)
p = None
default_arg = dict(DEFAULT_LAUNCHER_CONFIG)
args = load_launcher_config(defaults=default_arg)

p = None
dirPath = 'data/images'
characterList = []
studentModelList = []
studentModelCharacterMap = {}


_trt_cache_module = None


def _get_trt_cache_module():
    """Load cache helpers without importing CUDA or TensorRT."""
    global _trt_cache_module
    if _trt_cache_module is not None:
        return _trt_cache_module

    launcher_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = (
        os.path.join(launcher_dir, 'ezvtuber-rt', 'ezvtb_rt', 'trt_cache.py'),
        os.path.join(launcher_dir, 'ezvtuber-rt-main', 'ezvtb_rt', 'trt_cache.py'),
    )
    module_path = next((path for path in candidates if os.path.isfile(path)), None)
    if module_path is None:
        raise FileNotFoundError('找不到 ezvtb_rt/trt_cache.py')

    spec = importlib.util.spec_from_file_location(
        'easyvtuber_launcher_trt_cache', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'无法加载 TensorRT 缓存模块：{module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _trt_cache_module = module
    return module


def _format_cache_size(size_bytes):
    if size_bytes < 1024:
        return f'{size_bytes} B'
    if size_bytes < 1024 ** 2:
        return f'{size_bytes / 1024:.1f} KiB'
    if size_bytes < 1024 ** 3:
        return f'{size_bytes / 1024 ** 2:.1f} MiB'
    return f'{size_bytes / 1024 ** 3:.2f} GiB'


def is_nvidia_gpu():
    return has_nvidia_gpu()
hasTRTSupport = is_nvidia_gpu()

def refreshList():
    global characterList
    characterList = []
    for item in sorted(os.listdir(dirPath), key=lambda x: -os.path.getmtime(os.path.join(dirPath, x))):
        if '.png' == item[-4:]:
            characterList.append(item[:-4])


def scanStudentModels():
    """Scan custom_tha4_models directory for student models"""
    global studentModelList, studentModelCharacterMap
    custom_models_path = 'data/models/custom_tha4_models'
    studentModelList = scan_student_models(custom_models_path)
    studentModelCharacterMap = {
        model_name: os.path.join(
            custom_models_path,
            model_name,
            'character.png',
        )
        for model_name in studentModelList
    }


refreshList()
scanStudentModels()
dmlDeviceChoices, dmlDeviceMapping = launcher_directml_choices()

class OptionPanel(wx.Panel):
    def __init__(
        self,
        parent,
        title='',
        desc='',
        choices=None,
        mapping=None,
        type=0,
        default=None,
        disabled=False,
        mapper=min_cutoff_mapper,
        tooltip=None,
    ):
        wx.Panel.__init__(self, parent)
        self.type = type
        self.mapper = mapper
        if mapping is not None:
            self.mapping = mapping
        else:
            self.mapping = choices
        self.mainSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.leftSizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.mainSizer)
        self.titleText = wx.StaticText(self, wx.ID_ANY, title)
        titleFont = self.titleText.GetFont()
        titleFont.SetWeight(wx.FONTWEIGHT_SEMIBOLD)
        self.titleText.SetFont(titleFont)
        self.leftSizer.Add(self.titleText, 0, wx.ALL, 0)
        self.descText = wx.StaticText(self, wx.ID_ANY, desc)
        self.descText.Wrap(self.FromDIP(330))
        descFont = self.descText.GetFont()
        descFont.SetWeight(wx.FONTWEIGHT_EXTRALIGHT)
        self.descText.SetFont(descFont)
        self.leftSizer.Add(self.descText, 0, wx.TOP, self.FromDIP(2))
        self.mainSizer.Add(self.leftSizer, 1, wx.EXPAND | wx.ALL, 0)
        if self.type == 0:
            self.control = wx.Choice(self, wx.ID_ANY, choices=choices)
            self.control.SetMinSize(self.FromDIP(wx.Size(250, -1)))
            try:
                if default is not None:
                    if self.mapping:
                        self.control.SetSelection(self.mapping.index(default))
                    else:
                        self.control.SetSelection(default)
            except (ValueError, TypeError, IndexError):
                pass
            if self.control.GetSelection() == wx.NOT_FOUND and choices:
                self.control.SetSelection(0)
        elif self.type == 1:
            self.control = wx.CheckBox(self, wx.ID_ANY)
            try:
                if default is not None:
                    if self.mapping:
                        self.control.SetValue(self.mapping[default])
                    else:
                        self.control.SetValue(default)

            except (ValueError, TypeError, IndexError):
                pass
        elif self.type == 2:
            self.control = wx.TextCtrl(self, wx.ID_ANY)
            self.control.SetMinSize(self.FromDIP(wx.Size(250, -1)))
            try:
                if default is not None:
                    if self.mapping:
                        self.control.SetValue(self.mapping[default])
                    else:
                        self.control.SetValue(default)
            except (ValueError, TypeError, IndexError):
                pass
        elif self.type == 3:
            # Slider type for float values 0.0 to 1.0
            sliderSizer = wx.BoxSizer(wx.HORIZONTAL)
            self.control = wx.Slider(self, wx.ID_ANY, value=50, minValue=0, maxValue=100, 
                                    style=wx.SL_HORIZONTAL)
            self.control.SetMinSize(self.FromDIP(wx.Size(190, -1)))
            
            # Keep enough DPI-aware room for the longest formatted value
            # ("100.0000") and align changing numbers by their right edge.
            self.valueLabel = wx.StaticText(
                self,
                wx.ID_ANY,
                "0.0000",
                style=wx.ALIGN_RIGHT | wx.ST_NO_AUTORESIZE,
            )
            self.valueLabel.SetMinSize(self.FromDIP(wx.Size(80, -1)))
            valueFont = self.valueLabel.GetFont()
            valueFont.SetFamily(wx.FONTFAMILY_TELETYPE)
            self.valueLabel.SetFont(valueFont)
            
            try:
                if default is not None:
                    self.control.SetValue(default)
                    self.valueLabel.SetLabelText(f"{mapper(default):.4f}")
            except (ValueError, TypeError, IndexError):
                pass
            
            # Update label when slider changes
            def onSliderChange(event):
                val = mapper(self.control.GetValue())
                self.valueLabel.SetLabelText(f"{val:.4f}")
            self.control.Bind(wx.EVT_SLIDER, onSliderChange)
            
            sliderSizer.Add(self.control, 1, wx.ALIGN_CENTER_VERTICAL)
            sliderSizer.Add(self.valueLabel, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, self.FromDIP(10))
            self.mainSizer.Add(
                sliderSizer,
                0,
                wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT,
                self.FromDIP(12),
            )
            # Skip the normal control addition below
            self.control._slider_added = True

        if not (self.type == 3 and hasattr(self.control, '_slider_added')):
            self.mainSizer.Add(
                self.control,
                0,
                wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT,
                self.FromDIP(12),
            )
        if disabled:
            self.control.Enable(False)
        if tooltip:
            self.SetToolTip(tooltip)
            self.titleText.SetToolTip(tooltip)
            self.descText.SetToolTip(tooltip)
            self.control.SetToolTip(tooltip)

    def GetValue(self):
        if self.type == 0:
            ret = self.control.GetSelection()
        elif self.type == 1:
            ret = self.control.GetValue()
        elif self.type == 2:
            ret = self.control.GetValue()
        elif self.type == 3:
            ret = self.control.GetValue()
        if self.type == 0 and ret == wx.NOT_FOUND:
            return None
        if self.mapping is not None:
            return self.mapping[ret]
        else:
            return ret

    def SetValue(self, value):
        if self.type == 0:
            values = self.mapping or [
                self.control.GetString(i)
                for i in range(self.control.GetCount())
            ]
            try:
                self.control.SetSelection(values.index(value))
            except (ValueError, TypeError):
                if self.control.GetCount():
                    self.control.SetSelection(0)
        elif self.type == 1:
            self.control.SetValue(bool(value))
        elif self.type == 2:
            self.control.SetValue('' if value is None else str(value))
        elif self.type == 3:
            self.control.SetValue(int(value))
            self.valueLabel.SetLabelText(f'{self.mapper(int(value)):.4f}')

    def SetDescription(self, description):
        self.descText.SetLabelText(description)
        self.descText.Wrap(self.FromDIP(330))
        self.Layout()


def _important_log_line(line):
    """从 main 的一行日志中提取“重要”的简短描述，用于状态栏；无关行返回 None。"""
    line = line.strip()
    if not line:
        return None
    if line.startswith('Launched:'):
        return 'Launched'
    if 'Model Inference Ready' in line:
        return 'Model Inference Ready'
    # TRT: Building engine from ONNX: ...\filename.onnx
    if '[TRT]' in line and 'Building engine from ONNX:' in line:
        idx = line.find('Building engine from ONNX:')
        if idx != -1:
            path = line[idx + len('Building engine from ONNX:'):].strip().rstrip('\r\n')
            name = os.path.basename(path)
            if name:
                return f'Building: {name}'
    # TRT: Loading ONNX file from path ...\filename.onnx
    if '[TRT]' in line and 'Loading ONNX file from path' in line:
        idx = line.find('Loading ONNX file from path')
        if idx != -1:
            path = line[idx + len('Loading ONNX file from path'):].strip().strip('.').strip().rstrip('\r\n')
            name = os.path.basename(path)
            if name:
                return f'Loading: {name}'
    # ORT: Loading ONNX model from path ...\filename.onnx
    if '[ORT]' in line and 'Loading ONNX model from path' in line:
        idx = line.find('Loading ONNX model from path')
        if idx != -1:
            path = line[idx + len('Loading ONNX model from path'):].strip().strip('.').strip().rstrip('\r\n')
            name = os.path.basename(path)
            if name:
                return f'Loading: {name}'
    # ORT: Completed loading session: xxx.onnx
    if '[ORT]' in line and 'Completed loading session:' in line:
        idx = line.find('Completed loading session:')
        if idx != -1:
            name = line[idx + len('Completed loading session:'):].strip().rstrip('\r\n')
            if name:
                return f'Loaded: {name}'
    return None


def _on_main_log_line(panel, line):
    """在子线程中调用：若该行是重要日志，则用 wx.CallAfter 更新 panel 的状态框。"""
    display = _important_log_line(line)
    if display is not None:
        wx.CallAfter(panel.statusCtrl.SetValue, display)


def _read_pipe_to_stream(pipe, dest_stream, out_lines=None, on_line_callback=None):
    """从 pipe 读行，写回 dest_stream，可选追加到 out_lines，并对每行调用 on_line_callback(line)。"""
    if pipe is None:
        return
    try:
        for raw in iter(pipe.readline, b''):
            try:
                text = raw.decode('utf-8', errors='replace')
            except Exception:
                text = raw.decode('gbk', errors='replace')
            if out_lines is not None:
                out_lines.append(text)
            if on_line_callback is not None:
                on_line_callback(text)
            # 检查dest_stream是否可用（pythonw环境下可能不可用）
            if dest_stream is not None:
                try:
                    dest_stream.write(text)
                    dest_stream.flush()
                except (AttributeError, OSError, ValueError):
                    # pythonw环境下sys.stdout/sys.stderr可能不可用，忽略错误
                    pass
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


class PreviewCanvas(wx.Panel):
    """Aspect-fitted RGBA output drawn beside the launcher's settings."""

    def __init__(self, parent):
        super().__init__(parent, style=wx.BORDER_SIMPLE)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self._bitmap = None
        self._checkerboard = None
        self._checkerboard_size = 0
        self._message = '启动后在此显示角色画面'
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)

    def SetFrame(self, frame):
        height, width = frame.shape[:2]
        self._bitmap = wx.Bitmap.FromBufferRGBA(
            width,
            height,
            frame.tobytes(),
        )
        self.Refresh(False)

    def SetMessage(self, message, clear=False):
        self._message = message
        if clear:
            self._bitmap = None
        self.Refresh(False)

    def HasFrame(self):
        return self._bitmap is not None and self._bitmap.IsOk()

    def _get_checkerboard(self, size):
        if self._checkerboard is not None and self._checkerboard_size == size:
            return self._checkerboard

        bitmap = wx.Bitmap(size, size)
        dc = wx.MemoryDC(bitmap)
        dc.SetBackground(wx.Brush(wx.Colour(56, 56, 60)))
        dc.Clear()
        square = max(8, size // 16)
        dc.SetPen(wx.TRANSPARENT_PEN)
        colors = (wx.Colour(56, 56, 60), wx.Colour(76, 76, 82))
        row = 0
        current_y = 0
        while current_y < size:
            column = 0
            current_x = 0
            tile_height = min(square, size - current_y)
            while current_x < size:
                tile_width = min(square, size - current_x)
                dc.SetBrush(wx.Brush(colors[(row + column) % 2]))
                dc.DrawRectangle(
                    current_x,
                    current_y,
                    tile_width,
                    tile_height,
                )
                current_x += square
                column += 1
            current_y += square
            row += 1
        dc.SelectObject(wx.NullBitmap)
        self._checkerboard = bitmap
        self._checkerboard_size = size
        return bitmap

    def OnPaint(self, event):
        del event
        dc = wx.AutoBufferedPaintDC(self)
        dc.SetBackground(wx.Brush(wx.Colour(38, 38, 42)))
        dc.Clear()

        width, height = self.GetClientSize()
        padding = self.FromDIP(10)
        size = max(1, min(width - 2 * padding, height - 2 * padding))
        x = (width - size) // 2
        y = (height - size) // 2
        dc.DrawBitmap(self._get_checkerboard(size), x, y)

        if self.HasFrame():
            graphics = wx.GraphicsContext.Create(dc)
            if graphics is not None:
                graphics.DrawBitmap(self._bitmap, x, y, size, size)
            return

        dc.SetTextForeground(wx.Colour(225, 225, 230))
        font = dc.GetFont()
        font.SetWeight(wx.FONTWEIGHT_SEMIBOLD)
        dc.SetFont(font)
        dc.DrawLabel(
            self._message,
            wx.Rect(x, y, size, size),
            wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL,
        )

    def OnSize(self, event):
        self.Refresh(False)
        event.Skip()


class LauncherPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.number_of_buttons = 0
        self.frame = parent
        self.optionDict = {}
        self.optionSections = {}
        self.sectionPages = {}
        self.sectionSizers = {}
        self._applying_safety_preset = False
        self.previewBuffer = None
        self.previewFrameTimes = deque(maxlen=PREVIEW_FPS)
        self.previewStatusUpdatedAt = 0.0
        self.previewTimer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnPreviewTimer, self.previewTimer)
        self.main_output_lines = []   # main 的 stdout 副本
        self.main_stderr_lines = []   # main 的 stderr 副本
        self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        controlSizer = wx.BoxSizer(wx.HORIZONTAL)

        stEasy = wx.StaticText(self, wx.ID_ANY, "Easy")
        f = stEasy.GetFont()
        f.SetWeight(wx.FONTWEIGHT_HEAVY)
        f = f.MakeLarger()
        stEasy.SetFont(f)
        stVtuber = wx.StaticText(self, wx.ID_ANY, "Vtuber")
        f = stVtuber.GetFont()
        f.SetWeight(wx.FONTWEIGHT_LIGHT)
        f = f.MakeLarger()
        stVtuber.SetFont(f)
        controlSizer.Add(stEasy, 0, wx.ALL | wx.CENTER, 0)
        controlSizer.Add(stVtuber, 0, wx.RIGHT | wx.CENTER, 30)

        self.statusCtrl = wx.TextCtrl(
            self, wx.ID_ANY, '',
            style=wx.TE_READONLY | wx.BORDER_NONE | wx.TE_RIGHT,
        )
        self.statusCtrl.SetHint('当前操作')
        f = self.statusCtrl.GetFont()
        self.statusCtrl.SetFont(f.Smaller())
        controlSizer.Add(self.statusCtrl, 1, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 8)

        self.btnClearCache = wx.Button(self, label="清理 TRT 缓存…")
        self.btnClearCache.SetToolTip(
            '清理不会自动删除的 TensorRT 磁盘缓存；内存和显存缓存会在程序退出时自动释放。')
        self.btnClearCache.Bind(wx.EVT_BUTTON, self.OnClearTensorRTCache)
        controlSizer.Add(self.btnClearCache, 0, wx.CENTER | wx.ALL, 5)

        self.btnReset = wx.Button(self, label='恢复默认')
        self.btnReset.SetToolTip('先恢复界面默认值；只有点击“保存并启动”才会写入配置文件。')
        self.btnReset.Bind(wx.EVT_BUTTON, self.OnResetDefaults)
        controlSizer.Add(self.btnReset, 0, wx.CENTER | wx.ALL, 5)

        self.btnLaunch = wx.Button(self, label="保存并启动")
        self.btnLaunch.Bind(wx.EVT_BUTTON, self.OnLaunch)
        controlSizer.Add(self.btnLaunch, 0, wx.CENTER | wx.ALL, 10)

        self.mainSizer.Add(controlSizer, 0, wx.EXPAND | wx.LEFT, 10)
        self.mainSizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)
        self.SetSizer(self.mainSizer)

        self.contentSizer = wx.BoxSizer(wx.HORIZONTAL)

        self.previewPane = wx.Panel(self)
        self.previewPane.SetMinSize(self.FromDIP(wx.Size(300, -1)))
        previewSizer = wx.BoxSizer(wx.VERTICAL)
        self.previewPane.SetSizer(previewSizer)

        previewTitle = wx.StaticText(self.previewPane, label='角色输出预览')
        previewTitleFont = previewTitle.GetFont()
        previewTitleFont.SetWeight(wx.FONTWEIGHT_SEMIBOLD)
        previewTitleFont = previewTitleFont.MakeLarger()
        previewTitle.SetFont(previewTitleFont)
        previewSizer.Add(
            previewTitle,
            0,
            wx.LEFT | wx.RIGHT | wx.TOP,
            self.FromDIP(8),
        )

        self.previewCanvas = PreviewCanvas(self.previewPane)
        previewSizer.Add(
            self.previewCanvas,
            1,
            wx.EXPAND | wx.ALL,
            self.FromDIP(8),
        )
        self.previewStatusText = '未运行 · 启动后将在这里显示角色画面。'
        self.previewStatus = wx.StaticText(
            self.previewPane,
            label=self.previewStatusText,
        )
        self.previewStatus.Wrap(self.FromDIP(300))
        previewSizer.Add(
            self.previewStatus,
            0,
            wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM,
            self.FromDIP(8),
        )
        self.previewPane.Bind(wx.EVT_SIZE, self.OnPreviewPaneSize)

        self.notebook = wx.Notebook(self)
        self.notebook.SetMinSize(self.FromDIP(wx.Size(640, -1)))
        for section, label in (
            ('basic', '基本设置'),
            ('performance', '性能与安全'),
            ('advanced', '高级设置'),
        ):
            page = wx.ScrolledWindow(
                self.notebook,
                style=wx.VSCROLL | wx.TAB_TRAVERSAL,
            )
            page.SetScrollRate(0, self.FromDIP(12))
            page.SetBackgroundColour(self.GetBackgroundColour())
            page_sizer = wx.BoxSizer(wx.VERTICAL)
            page.SetSizer(page_sizer)
            self.notebook.AddPage(page, label)
            self.sectionPages[section] = page
            self.sectionSizers[section] = page_sizer

        self.contentSizer.Add(
            self.previewPane,
            2,
            wx.EXPAND | wx.LEFT | wx.TOP | wx.BOTTOM,
            self.FromDIP(8),
        )
        self.contentSizer.Add(
            wx.StaticLine(self, style=wx.LI_VERTICAL),
            0,
            wx.EXPAND | wx.ALL,
            self.FromDIP(8),
        )
        self.contentSizer.Add(
            self.notebook,
            3,
            wx.EXPAND | wx.TOP | wx.RIGHT | wx.BOTTOM,
            self.FromDIP(8),
        )
        self.mainSizer.Add(self.contentSizer, 1, wx.EXPAND)

        def addOption(key, section='basic', **kwargs):
            kwargs['default'] = args[key]
            parent_page = self.sectionPages[section]
            t = OptionPanel(parent_page, **kwargs)
            self.sectionSizers[section].Add(
                t,
                0,
                wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP,
                self.FromDIP(8),
            )
            self.optionDict[key] = t
            self.optionSections[key] = section
            return t

        addOption(
            'character',
            title='角色 / Character',
            desc='选择 data/images 中的角色图片。',
            choices=characterList,
            tooltip='THA4 Student 模型自带角色图，选择该模型后此项会自动锁定。',
        )

        addOption(
            'input',
            title='输入 / Input',
            desc='选择面捕数据源。',
            choices=[
                'iFacialMocap',
                'OpenSeeFace',
                'OpenCV（摄像头）',
                '鼠标输入',
                '调试输入',
            ],
            mapping=[0, 4, 1, 3, 2],
        )
        addOption(
            'ifm',
            title='iFacialMocap 地址',
            desc='填写手机显示的 IP；未写端口时使用 49983。',
            type=2,
        )
        addOption(
            'is_eyebrow',
            title='眉毛追踪',
            desc='使用眉毛输入，会略微增加推理开销。',
            type=1,
            default=True,
        )
        addOption(
            'osf',
            title='OpenSeeFace 地址',
            desc='填写 IP:端口，默认 127.0.0.1:11573。',
            type=2,
        )
        addOption(
            'mouse_audio_input',
            title='音频驱动口型',
            desc='鼠标输入模式下，用 WASAPI 音量控制嘴部。',
            type=1,
        )
        addOption(
            'audio_sensitivity',
            title='音频灵敏度',
            desc='控制音量对嘴部动作的影响程度。',
            type=2,
        )
        addOption(
            'audio_threshold',
            title='音频阈值',
            desc='低于此值的音量会被忽略。',
            type=2,
        )
        addOption(
            'blink_interval',
            title='眨眼间隔',
            desc='鼠标输入模式下的自动眨眼间隔。',
            choices=['关闭', '3 秒', '5 秒', '7 秒'],
            mapping=['inf', '3.0', '5.0', '7.0'],
        )
        addOption(
            'breath_cycle',
            title='呼吸循环',
            desc='自动呼吸间隔；启用后会增加少量占用。',
            choices=['关闭', '3 秒', '5 秒', '7 秒'],
            mapping=['inf', '3.0', '5.0', '7.0'],
        )
        addOption(
            'output',
            title='输出 / Output',
            desc='三种模式都会在左侧预览；前两项同时向外部程序发送。',
            choices=['Spout2（OBS）', 'OBS VirtualCam', '仅启动器窗口'],
            mapping=[0, 1, 2],
        )
        addOption(
            'safety_preset',
            section='performance',
            title='运行安全预设',
            desc='只组合下方两项；也可直接修改任一项进入自定义。',
            choices=[
                '保守（低温优先）',
                '平衡（推荐）',
                '性能（高吞吐）',
                '自定义（调整下方两项）',
            ],
            mapping=['Conservative', 'Balanced', 'Performance', 'Custom'],
            tooltip='手动修改帧率或 GPU 持续占空目标后会自动显示为“自定义”。',
        )

        self.performanceNotice = wx.StaticText(
            self.sectionPages['performance'],
            label='',
        )
        notice_font = self.performanceNotice.GetFont()
        notice_font.SetWeight(wx.FONTWEIGHT_SEMIBOLD)
        self.performanceNotice.SetFont(notice_font)
        self.performanceNotice.Wrap(self.FromDIP(620))
        self.sectionSizers['performance'].Add(
            self.performanceNotice,
            0,
            wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP,
            self.FromDIP(8),
        )

        addOption(
            'frame_rate_limit',
            section='performance',
            title='输出帧率',
            desc='与安全预设放在一起；本机温度敏感时建议先用 24 FPS。',
            choices=['10', '15', '20', '24', '30', '60'],
        )
        addOption(
            'gpu_duty_limit',
            section='performance',
            title='GPU 持续占空目标',
            desc='限制连续推理时间；不是温度或瞬时利用率上限。',
            choices=['70%', '80%', '90%', '95%', '100%（关闭限制）'],
            mapping=['70', '80', '90', '95', '100'],
            tooltip='一次模型推理或 TensorRT 引擎构建仍可能瞬时达到 100%。',
        )
        addOption(
            'use_tensorrt',
            section='performance',
            title='TensorRT 后端',
            desc='仅 NVIDIA；关闭时使用 DirectML。',
            type=1,
            tooltip='TensorRT 首次使用或缓存不匹配时需要构建引擎，可能出现短时高 GPU 负载。',
        )
        addOption(
            'dml_device',
            section='performance',
            title='DirectML 显卡',
            desc='Auto（推荐）优先选择高性能独显。',
            choices=dmlDeviceChoices,
            mapping=dmlDeviceMapping,
            tooltip='显式编号用于排查或严格复现；不同显卡可能产生少量 8-bit 舍入差异。',
        )
        addOption(
            'ram_cache_size',
            section='performance',
            title='RAM 最终帧缓存',
            desc='普通帧与超分帧共享的总预算。',
            choices=['关闭', '1 GiB', '2 GiB', '4 GiB', '8 GiB', '16 GiB'],
            mapping=['0b', '1gb', '2gb', '4gb', '8gb', '16gb'],
        )
        addOption(
            'ram_cache_mode',
            section='performance',
            title='RAM 缓存存储模式',
            desc='Raw 推荐低延迟；Brotli 推荐节省内存。',
            choices=['Raw（低延迟）', 'Brotli（省内存）'],
            mapping=['raw', 'brotli'],
            tooltip='两种模式均无损。Raw 命中无需同步解压；Brotli 能容纳更多姿态，但命中有解压开销。',
        )
        addOption(
            'vram_cache_size',
            section='performance',
            title='VRAM 模型中间缓存',
            desc='仅 TensorRT 生效，受严格总预算限制。',
            choices=['关闭', '1 GiB', '2 GiB', '4 GiB', '8 GiB', '16 GiB'],
            mapping=['0b', '1gb', '2gb', '4gb', '8gb', '16gb'],
        )
        addOption(
            'cache_simplify',
            section='performance',
            title='输入简化',
            desc='级别越高，缓存命中越多，但动作阶梯感也越明显。',
            choices=['Off', 'Low', 'Medium', 'High', 'Higher', 'Highest', 'Gaming'],
            tooltip='关闭输入简化时最终帧 RAM 缓存不会启用。Medium 是当前画质与命中率的推荐折中。',
        )

        # Build model_select choices
        model_choices = ['Seperable Half', 'Seperable Full', 'Standard Half',
                         'Standard Full', 'THA4 Half', 'THA4 Full']
        model_mapping = ['seperable_half', 'seperable_full',
                         'standard_half', 'standard_full', 'tha4_half',
                         'tha4_full']

        # Add student models if available
        for student_model in studentModelList:
            model_choices.append(f'THA4 Student ({student_model})')
            model_mapping.append(f'tha4_student_{student_model}')

        addOption(
            'model_select',
            section='advanced',
            title='模型与精度',
            desc='独立选择；运行安全预设不会修改此项。',
            choices=model_choices,
            mapping=model_mapping,
            tooltip='Standard Full 精度较高、性能较低；Half 通常更快，但兼容性取决于显卡。',
        )
        addOption(
            'sr',
            section='advanced',
            title='超分辨率',
            desc='关闭时输出 512²；开启后输出 1024² 并增加负载。',
            choices=[
                'Off',
                'anime4k_x2',
                'waifu2x_x2_half',
                'real-esrgan_x4_half',
                'waifu2x_x2_full',
                'real-esrgan_x4_full',
            ],
        )
        addOption(
            'interpolation',
            section='advanced',
            title='RIFE 补帧',
            desc='可降低同等输出帧率下的 THA 推理次数，但会增加延迟。',
            choices=['Off', 'x2_half', 'x3_half', 'x4_half', 'x2_full', 'x3_full', 'x4_full'],
        )
        addOption(
            'min_cutoff',
            section='advanced',
            title='滤波 Min Cutoff',
            desc='越小越平滑，越大静止时越灵敏。',
            type=3,
            mapper=min_cutoff_mapper,
        )
        addOption(
            'beta',
            section='advanced',
            title='滤波 Beta',
            desc='越小越平滑，越大运动时越灵敏。',
            type=3,
            mapper=beta_mapper,
        )
        addOption(
            'is_alpha_clean',
            section='advanced',
            title='Alpha 预处理',
            desc='清理透明区域 RGB，代替部分蒙版后处理。',
            type=1,
        )
        addOption(
            'is_extend_movement',
            section='advanced',
            title='扩展移动',
            desc='根据面捕 XY 输入移动和缩放角色。',
            type=1,
        )
        addOption(
            'is_bongo',
            section='advanced',
            title='Bongocat 模式',
            desc='旋转输出以适配 Bongocat 桌宠。',
            type=1,
        )
        addOption(
            'is_alpha_split',
            section='advanced',
            title='Alpha Split',
            desc='仅旧 VirtualCam 工作流需要；Spout2 已原生支持透明通道。',
            type=1,
        )

        self.optionDict['input'].control.Bind(
            wx.EVT_CHOICE, self.OnInputChoice)
        self.optionDict['mouse_audio_input'].control.Bind(
            wx.EVT_CHECKBOX, self.OnAudioInputChoice)
        self.optionDict['safety_preset'].control.Bind(
            wx.EVT_CHOICE, self.OnSafetyPresetChoice)
        self.optionDict['frame_rate_limit'].control.Bind(
            wx.EVT_CHOICE, self.OnPacingSettingChoice)
        self.optionDict['gpu_duty_limit'].control.Bind(
            wx.EVT_CHOICE, self.OnPacingSettingChoice)
        self.optionDict['model_select'].control.Bind(
            wx.EVT_CHOICE, self.OnModelSelect)
        self.optionDict['use_tensorrt'].control.Bind(
            wx.EVT_CHECKBOX, self.OnBackendChoice)
        self.optionDict['ram_cache_size'].control.Bind(
            wx.EVT_CHOICE, self.OnCacheSettingsChanged)
        self.optionDict['ram_cache_mode'].control.Bind(
            wx.EVT_CHOICE, self.OnCacheSettingsChanged)
        self.optionDict['cache_simplify'].control.Bind(
            wx.EVT_CHOICE, self.OnCacheSettingsChanged)
        self.optionDict['sr'].control.Bind(
            wx.EVT_CHOICE, self.OnCacheSettingsChanged)
        self.optionDict['output'].control.Bind(
            wx.EVT_CHOICE, self.OnOutputChoice)
        self.notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnNotebookPageChanged)

        if not hasTRTSupport:
            self.optionDict['use_tensorrt'].control.SetValue(False)
            self.optionDict['use_tensorrt'].control.Enable(False)
            self.optionDict['use_tensorrt'].control.SetToolTip(
                '需要NVIDIA显卡支持才能使用TensorRT')

        self.OnInputChoice(layout=False)
        self.OnModelSelect()
        self.OnBackendChoice(layout=False)
        self.OnCacheSettingsChanged(layout=False)
        self.OnOutputChoice(layout=False)
        self._update_safety_notice()
        self._layout_options()
        self.frame.Bind(wx.EVT_ACTIVATE, self.OnActivate)

    def _show_option(self, key, visible):
        self.optionDict[key].Show(bool(visible))

    def _layout_options(self):
        for page in self.sectionPages.values():
            page.Layout()
            page.FitInside()
        self.Layout()
        self.frame.Layout()

    def _set_preview_status(self, text):
        self.previewStatusText = text
        self._wrap_preview_status()
        self.previewPane.Layout()

    def _wrap_preview_status(self):
        self.previewStatus.SetLabelText(self.previewStatusText)
        self.previewStatus.Wrap(max(
            self.FromDIP(220),
            self.previewPane.GetClientSize().width - self.FromDIP(16),
        ))

    def OnPreviewPaneSize(self, event):
        self._wrap_preview_status()
        self.previewPane.Layout()
        event.Skip()

    def StartPreview(self):
        self.StopPreview(clear=True)
        try:
            self.previewBuffer = PreviewSharedBuffer.create()
        except (OSError, MemoryError, ValueError) as error:
            self.previewCanvas.SetMessage('内嵌预览不可用', clear=True)
            self._set_preview_status(
                f'预览初始化失败；外部输出仍可继续：{error}'
            )
            return None

        self.previewFrameTimes.clear()
        self.previewStatusUpdatedAt = 0.0
        self.previewCanvas.SetMessage('正在等待角色画面…', clear=True)
        self._set_preview_status(
            f'正在启动 · 预览最高 {PREVIEW_FPS} FPS，不限制实际输出。'
        )
        self.previewTimer.Start(max(1, round(1000 / PREVIEW_FPS)))
        return self.previewBuffer.name

    def StopPreview(self, message='未运行', clear=True):
        if self.previewTimer.IsRunning():
            self.previewTimer.Stop()
        if self.previewBuffer is not None:
            self.previewBuffer.close()
            self.previewBuffer = None
        self.previewFrameTimes.clear()
        self.previewCanvas.SetMessage(message, clear=clear)
        self._set_preview_status(message)

    def OnPreviewTimer(self, event=None):
        global p
        if p is not None:
            return_code = p.poll()
            if return_code is not None:
                p = None
                self.btnLaunch.SetLabelText('保存并启动')
                if return_code == 0:
                    message = '运行已结束'
                else:
                    message = f'运行异常退出（代码 {return_code}）'
                self.statusCtrl.SetValue(message)
                self.StopPreview(message=message, clear=True)
                return

        if self.previewBuffer is None:
            return
        try:
            frame = self.previewBuffer.read_latest()
        except (OSError, TypeError, ValueError) as error:
            self.previewTimer.Stop()
            self.previewCanvas.SetMessage('预览读取失败', clear=True)
            self._set_preview_status(
                f'预览读取失败；实际输出仍在运行：{error}'
            )
            return
        if frame is None:
            return

        self.previewCanvas.SetFrame(frame)
        now = time.perf_counter()
        self.previewFrameTimes.append(now)
        if now - self.previewStatusUpdatedAt >= 0.5:
            if len(self.previewFrameTimes) >= 2:
                elapsed = self.previewFrameTimes[-1] - self.previewFrameTimes[0]
                preview_fps = (
                    (len(self.previewFrameTimes) - 1) / elapsed
                    if elapsed > 0 else 0.0
                )
                status = (
                    f'运行中 · 预览 {preview_fps:.1f} FPS · '
                    '不限制实际输出。'
                )
            else:
                status = '运行中 · 已收到首帧 · 不限制实际输出。'
            self._set_preview_status(status)
            self.previewStatusUpdatedAt = now

        if event is not None:
            event.Skip()

    def _update_safety_notice(self):
        preset = self.optionDict['safety_preset'].GetValue()
        frame_rate = self.optionDict['frame_rate_limit'].GetValue()
        duty = self.optionDict['gpu_duty_limit'].GetValue()
        labels = {
            'Conservative': '保守模式：优先降低长时间温度与占用。',
            'Balanced': '平衡模式：30 FPS，并为其他程序保留更多 GPU 时间。',
            'Performance': '性能模式：提高持续推理预算，请留意温度。',
            'Custom': '自定义模式：请直接调整下面的“输出帧率”和“GPU 持续占空目标”。',
        }
        self.performanceNotice.SetLabelText(
            f'{labels.get(preset, labels["Custom"])} 当前 {frame_rate} FPS / '
            f'{duty}% 持续占空目标。此目标不会限制单次推理或引擎构建的瞬时峰值。'
        )
        self.performanceNotice.Wrap(self.FromDIP(620))

    def OnInputChoice(self, event=None, layout=True):
        input_mode = self.optionDict['input'].GetValue()
        self._show_option('ifm', input_mode == 0)
        self._show_option('osf', input_mode == 4)
        self._show_option('is_eyebrow', input_mode in (0, 4))
        self._show_option('min_cutoff', input_mode in (1, 4))
        self._show_option('beta', input_mode in (1, 4))
        mouse_input = input_mode == 3
        self._show_option('mouse_audio_input', mouse_input)
        self._show_option('blink_interval', mouse_input)
        self.OnAudioInputChoice(layout=False)
        if layout:
            self._layout_options()
        if event is not None:
            event.Skip()

    def OnAudioInputChoice(self, event=None, layout=True):
        visible = (
            self.optionDict['input'].GetValue() == 3
            and self.optionDict['mouse_audio_input'].GetValue()
        )
        self._show_option('audio_sensitivity', visible)
        self._show_option('audio_threshold', visible)
        if layout:
            self._layout_options()
        if event is not None:
            event.Skip()

    def OnSafetyPresetChoice(self, event=None):
        preset = self.optionDict['safety_preset'].GetValue()
        if preset in SAFETY_PRESETS:
            self._applying_safety_preset = True
            try:
                values = SAFETY_PRESETS[preset]
                self.optionDict['frame_rate_limit'].SetValue(
                    values['frame_rate_limit'])
                self.optionDict['gpu_duty_limit'].SetValue(
                    values['gpu_duty_limit'])
            finally:
                self._applying_safety_preset = False
        self._update_safety_notice()
        self._layout_options()
        if event is not None:
            event.Skip()

    def OnPacingSettingChoice(self, event=None):
        if not self._applying_safety_preset:
            preset = infer_safety_preset(
                self.optionDict['frame_rate_limit'].GetValue(),
                self.optionDict['gpu_duty_limit'].GetValue(),
            )
            self.optionDict['safety_preset'].SetValue(preset)
        self._update_safety_notice()
        self._layout_options()
        if event is not None:
            event.Skip()

    def OnModelSelect(self, event=None):
        model_value = self.optionDict['model_select'].GetValue() or ''
        char_ctrl = self.optionDict['character']
        is_student_model = 'tha4_student_' in model_value
        char_ctrl.control.Enable(not is_student_model)
        if is_student_model:
            tooltip = '已锁定：THA4 Student 模型包含自己的角色图片。'
        else:
            tooltip = '选择 data/images 中的角色图片。'
        char_ctrl.control.SetToolTip(tooltip)
        if event is not None:
            event.Skip()

    def OnActivate(self, event):
        global characterList
        if not event.GetActive():
            event.Skip()
            return
        char_ctrl = self.optionDict['character'].control
        current_name = (
            char_ctrl.GetStringSelection()
            if char_ctrl.GetSelection() >= 0
            else ''
        )
        refreshList()
        scanStudentModels()
        self.optionDict['character'].mapping = characterList
        char_ctrl.SetItems(characterList)
        try:
            char_ctrl.SetSelection(characterList.index(current_name))
        except (ValueError, TypeError):
            if characterList:
                char_ctrl.SetSelection(0)
        self.OnModelSelect()
        event.Skip()

    def OnBackendChoice(self, event=None, layout=True):
        using_tensorrt = self.optionDict['use_tensorrt'].GetValue()
        self._show_option('dml_device', not using_tensorrt)
        self._show_option('vram_cache_size', using_tensorrt)
        if layout:
            self._layout_options()
        if event is not None:
            event.Skip()

    def OnCacheSettingsChanged(self, event=None, layout=True):
        cache_size = self.optionDict['ram_cache_size'].GetValue()
        storage_mode = self.optionDict['ram_cache_mode'].GetValue()
        simplify_enabled = self.optionDict['cache_simplify'].GetValue() != 'Off'
        super_resolution = self.optionDict['sr'].GetValue() != 'Off'
        description = describe_ram_cache(
            cache_size,
            storage_mode,
            super_resolution=super_resolution,
            simplify_enabled=simplify_enabled,
        )
        self.optionDict['ram_cache_size'].SetDescription(description)
        self._show_option(
            'ram_cache_mode',
            cache_size != '0b' and simplify_enabled,
        )
        if layout:
            self._layout_options()
        if event is not None:
            event.Skip()

    def OnOutputChoice(self, event=None, layout=True):
        self._show_option(
            'is_alpha_split',
            self.optionDict['output'].GetValue() != 0,
        )
        if layout:
            self._layout_options()
        if event is not None:
            event.Skip()

    def OnNotebookPageChanged(self, event):
        self._layout_options()
        event.Skip()

    def ApplyDefaultSettings(self):
        for key, option in self.optionDict.items():
            if key in DEFAULT_LAUNCHER_CONFIG:
                option.SetValue(DEFAULT_LAUNCHER_CONFIG[key])
        if not hasTRTSupport:
            self.optionDict['use_tensorrt'].SetValue(False)
        self.OnInputChoice(layout=False)
        self.OnModelSelect()
        self.OnBackendChoice(layout=False)
        self.OnCacheSettingsChanged(layout=False)
        self.OnOutputChoice(layout=False)
        self.OnPacingSettingChoice()
        self.notebook.SetSelection(0)
        self.statusCtrl.SetValue('已恢复默认值（尚未保存）')
        self._layout_options()

    def OnResetDefaults(self, event):
        del event
        answer = wx.MessageBox(
            '将界面恢复为推荐默认值。\n\n'
            '当前配置不会立即写入；点击“保存并启动”后才会保存。\n\n'
            '确定继续吗？',
            '恢复默认值',
            wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION,
            self,
        )
        if answer == wx.YES:
            self.ApplyDefaultSettings()

    def ConfirmTensorRTStartup(self, gpu_duty_limit):
        """Require an explicit acknowledgement before TensorRT can start."""
        try:
            cache = _get_trt_cache_module()
            cache_dir = cache.get_cache_dir().resolve()
            lock_files = cache.list_active_cache_locks(cache_dir)
            engine_files = [
                path
                for path in cache.list_cache_files(cache_dir)
                if path.name.endswith('.trt')
            ]
            engine_bytes = 0
            for path in engine_files:
                try:
                    engine_bytes += path.stat().st_size
                except FileNotFoundError:
                    pass
        except Exception as error:
            wx.MessageBox(
                '无法确认 TensorRT 缓存状态，因此已取消启动。\n\n'
                f'{error}',
                'TensorRT 启动保护',
                wx.OK | wx.ICON_ERROR,
                self,
            )
            return False

        if lock_files:
            wx.MessageBox(
                '检测到 TensorRT 引擎构建锁。为避免同时构建或误删缓存，已取消启动。\n\n'
                f'缓存位置：{cache_dir}\n'
                f'锁文件：{lock_files[0].name}',
                'TensorRT 缓存正在使用',
                wx.OK | wx.ICON_WARNING,
                self,
            )
            return False

        cache_size = _format_cache_size(engine_bytes)
        if engine_files:
            cache_summary = (
                f'当前目录中找到 {len(engine_files)} 个 TensorRT 引擎缓存'
                f'（约 {cache_size}）。\n'
                '若所选模型、精度、显卡或 TensorRT 版本与缓存不匹配，'
                '仍会重新构建所需引擎。'
            )
        else:
            cache_summary = (
                '当前目录中没有 TensorRT 引擎缓存。\n'
                '本次启动将构建所选配置需要的引擎。'
            )

        answer = wx.MessageBox(
            '即将启动 TensorRT。\n\n'
            f'{cache_summary}\n\n'
            f'缓存位置：{cache_dir}\n\n'
            '引擎构建是不可细分的单次操作，期间 GPU 可能瞬时达到很高负载。\n'
            f'当前 {gpu_duty_limit}% 持续占空比限制会在构建后冷却，'
            '但不能硬性限制构建中的瞬时峰值。\n\n'
            '确定继续吗？',
            'TensorRT 启动确认',
            wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING,
            self,
        )
        return answer == wx.YES

    def OnClearTensorRTCache(self, e):
        del e
        global p

        if p is not None and p.poll() is None:
            wx.MessageBox(
                '请先用启动器停止 EasyVtuber，再清理 TensorRT 缓存。',
                '无法清理缓存',
                wx.OK | wx.ICON_WARNING,
                self,
            )
            return

        try:
            cache = _get_trt_cache_module()
            # Ensure older %TEMP% caches are safely migrated before inspection.
            cache_dir = cache.get_cache_dir().resolve()
            lock_files = cache.list_active_cache_locks(cache_dir)
            file_count, total_bytes = cache.get_cache_usage(cache_dir)
        except Exception as error:
            wx.MessageBox(
                f'读取 TensorRT 缓存失败：\n{error}',
                '缓存检查失败',
                wx.OK | wx.ICON_ERROR,
                self,
            )
            return

        if lock_files:
            wx.MessageBox(
                '检测到 TensorRT 引擎正在构建或遗留了构建锁，未删除任何文件。\n\n'
                f'缓存位置：{cache_dir}\n'
                f'锁文件：{lock_files[0].name}',
                '缓存正在使用',
                wx.OK | wx.ICON_WARNING,
                self,
            )
            return

        if file_count == 0:
            self.statusCtrl.SetValue('没有可清理的 TRT 磁盘缓存')
            wx.MessageBox(
                '没有找到可清理的 TensorRT 磁盘缓存。\n\n'
                'RAM 和显存缓存会在 EasyVtuber 退出时自动释放，无需手动清理。\n\n'
                f'检查位置：{cache_dir}',
                '无需清理',
                wx.OK | wx.ICON_INFORMATION,
                self,
            )
            return

        answer = wx.MessageBox(
            f'将删除 {file_count} 个 TensorRT 磁盘缓存文件'
            f'（约 {_format_cache_size(total_bytes)}）。\n\n'
            f'缓存位置：{cache_dir}\n\n'
            '下次启用 TensorRT 时必须重新构建引擎，构建期间可能出现短时很高的 GPU 负载。\n'
            '除非缓存损坏或需要释放磁盘空间，否则不建议清理。\n\n'
            '确定继续吗？',
            '清理 TensorRT 缓存',
            wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING,
            self,
        )
        if answer != wx.YES:
            return

        try:
            deleted_count, deleted_bytes = cache.clear_cache(cache_dir)
        except cache.CacheInUseError as error:
            wx.MessageBox(
                f'缓存已开始被引擎构建使用，未继续清理：\n{error}',
                '缓存正在使用',
                wx.OK | wx.ICON_WARNING,
                self,
            )
            return
        except Exception as error:
            wx.MessageBox(
                f'清理 TensorRT 缓存失败：\n{error}',
                '清理失败',
                wx.OK | wx.ICON_ERROR,
                self,
            )
            return

        display_size = _format_cache_size(deleted_bytes)
        self.statusCtrl.SetValue(
            f'已清理 {deleted_count} 个 TRT 缓存文件（{display_size}）')
        wx.MessageBox(
            f'已清理 {deleted_count} 个 TensorRT 缓存文件，共 {display_size}。\n\n'
            '下次启用 TensorRT 时会重新构建引擎。',
            '清理完成',
            wx.OK | wx.ICON_INFORMATION,
            self,
        )

    def OnLaunch(self, e):
        del e
        global p
        settings = {
            key: option.GetValue()
            for key, option in self.optionDict.items()
        }
        try:
            settings = save_launcher_config(settings)
        except OSError as error:
            self.statusCtrl.SetValue('保存配置失败')
            wx.MessageBox(
                f'无法保存 launcher.json：\n{error}',
                '配置保存失败',
                wx.OK | wx.ICON_ERROR,
                self,
            )
            return

        self.btnLaunch.SetLabelText('处理中…')

        if p is not None and p.poll() is None:
            creation_flags = 0
            if sys.platform == 'win32':
                # CREATE_NO_WINDOW = 0x08000000
                creation_flags = 0x08000000
            subprocess.run(['taskkill', '/F', '/PID', str(p.pid), '/T'], 
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL,
                          creationflags=creation_flags)
            p = None
            self.StopPreview(message='已停止', clear=True)
            self.statusCtrl.Clear()
            self.btnLaunch.SetLabelText('保存并启动')
            return

        p = None
        if settings['use_tensorrt'] and not self.ConfirmTensorRTStartup(
                settings['gpu_duty_limit']):
            self.statusCtrl.SetValue('已取消 TensorRT 启动')
            self.btnLaunch.SetLabelText('保存并启动')
            return

        preview_shm_name = self.StartPreview()

        # If the launcher itself uses pythonw, start main with python.exe so its
        # output can still be captured by the status reader.
        python_exe = sys.executable
        if 'pythonw' in python_exe.lower():
            python_exe = python_exe.replace(
                'pythonw.exe', 'python.exe').replace('pythonw', 'python')
        display_size = wx.GetDisplaySize()
        run_args = build_launch_command(
            settings,
            python_exe,
            (display_size.width, display_size.height),
            preview_shm_name=preview_shm_name,
        )

        print('Launched: ' + ' '.join(run_args))
        self.main_output_lines.clear()
        self.main_stderr_lines.clear()
        self.statusCtrl.SetValue('正在启动')
        on_line = lambda line: _on_main_log_line(self, line)
        creation_flags = 0
        if sys.platform == 'win32':
            # CREATE_NO_WINDOW = 0x08000000
            creation_flags = 0x08000000
        try:
            p = subprocess.Popen(
                run_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creation_flags,
            )
        except OSError as error:
            self.StopPreview(message='启动失败', clear=True)
            self.statusCtrl.SetValue('启动失败')
            self.btnLaunch.SetLabelText('保存并启动')
            wx.MessageBox(
                f'无法启动 EasyVtuber：\n{error}',
                '启动失败',
                wx.OK | wx.ICON_ERROR,
                self,
            )
            return

        threading.Thread(
            target=_read_pipe_to_stream,
            args=(p.stdout, sys.stdout, self.main_output_lines, on_line),
            daemon=True,
        ).start()
        threading.Thread(
            target=_read_pipe_to_stream,
            args=(p.stderr, sys.stderr, self.main_stderr_lines, on_line),
            daemon=True,
        ).start()
        self.btnLaunch.SetLabelText('停止')


class MainFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MainFrame, self).__init__(*args, **kw)
        self.InitUi()

        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def OnClose(self, e):
        global p
        if p is not None:
            creation_flags = 0
            if sys.platform == 'win32':
                # CREATE_NO_WINDOW = 0x08000000
                creation_flags = 0x08000000
            subprocess.run(['taskkill', '/F', '/PID', str(p.pid), '/T'], 
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL,
                          creationflags=creation_flags)
            p = None
        self.panel.StopPreview(message='已关闭', clear=True)
        e.Skip()

    def InitUi(self):
        self.SetTitle("EasyVtuber Launcher")
        self.fSizer = wx.BoxSizer(wx.VERTICAL)
        self.panel = LauncherPanel(self)
        self.fSizer.Add(self.panel, 1, wx.EXPAND)
        self.SetSizer(self.fSizer)
        self.SetMinSize(self.FromDIP(wx.Size(1120, 520)))
        self.SetClientSize(self.FromDIP(wx.Size(1180, 700)))
        self.Layout()
        self.panel._layout_options()
        self.Centre()


def main():
    app = wx.App()
    sample = MainFrame(None)

    sample.Show()
    app.MainLoop()


if __name__ == "__main__":
    main()
