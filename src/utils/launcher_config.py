import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path


LAUNCHER_CONFIG_VERSION = 2

CACHE_SIMPLIFY_MAP = {
    'Off': 0,
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Higher': 4,
    'Highest': 6,
    'Gaming': 8,
}

# These presets only combine pacing controls. They deliberately do not change
# model precision, pose simplification, interpolation, SR, or cache capacity.
SAFETY_PRESETS = {
    'Conservative': {
        'frame_rate_limit': '24',
        'gpu_duty_limit': '70',
    },
    'Balanced': {
        'frame_rate_limit': '30',
        'gpu_duty_limit': '80',
    },
    'Performance': {
        'frame_rate_limit': '30',
        'gpu_duty_limit': '90',
    },
}

DEFAULT_LAUNCHER_CONFIG = {
    'config_version': LAUNCHER_CONFIG_VERSION,
    'character': 'lambda_00',
    'input': 3,
    'output': 2,
    'ifm': None,
    'osf': '127.0.0.1:11573',
    'min_cutoff': 50,
    'beta': 80,
    'is_extend_movement': False,
    'is_alpha_split': False,
    'is_bongo': False,
    'is_alpha_clean': False,
    'is_eyebrow': False,
    'cache_simplify': 'High',
    'ram_cache_size': '2gb',
    'ram_cache_mode': 'raw',
    'vram_cache_size': '2gb',
    'model_select': 'seperable_half',
    'interpolation': 'Off',
    'frame_rate_limit': '30',
    'gpu_duty_limit': '80',
    'sr': 'Off',
    'use_tensorrt': False,
    'dml_device': 'auto',
    'safety_preset': 'Balanced',
    'mouse_audio_input': False,
    'audio_sensitivity': '0.02',
    'audio_threshold': '10.0',
    'blink_interval': '5.0',
    'breath_cycle': 'inf',
}


def min_cutoff_mapper(value, revert=False):
    """Map the launcher slider's integer range to the filter cutoff."""
    if revert:
        return int((value / 100.0) ** 0.5 * 100)
    return (value / 100.0) ** 2 * 100.0


def beta_mapper(value, revert=False):
    """Map the launcher slider's integer range to the filter beta."""
    if revert:
        return int((value ** 0.5) * 100)
    return (value / 100.0) ** 2


def infer_safety_preset(frame_rate_limit, gpu_duty_limit):
    """Return the matching pacing preset, or Custom for an exact custom pair."""
    frame_rate_limit = str(frame_rate_limit)
    gpu_duty_limit = str(gpu_duty_limit)
    for name, values in SAFETY_PRESETS.items():
        if (
            values['frame_rate_limit'] == frame_rate_limit
            and values['gpu_duty_limit'] == gpu_duty_limit
        ):
            return name
    return 'Custom'


def apply_safety_preset(config, preset_name):
    """Return a copy with only the selected pacing controls changed."""
    if preset_name != 'Custom' and preset_name not in SAFETY_PRESETS:
        raise ValueError(f'Unknown safety preset: {preset_name}')

    updated = dict(config)
    if preset_name in SAFETY_PRESETS:
        updated.update(SAFETY_PRESETS[preset_name])
    updated['safety_preset'] = preset_name
    return updated


def normalize_launcher_config(saved=None, defaults=None):
    """Load old/new launcher settings without letting legacy presets alter quality."""
    normalized = dict(
        DEFAULT_LAUNCHER_CONFIG if defaults is None else defaults
    )
    saved_mapping = saved if isinstance(saved, Mapping) else {}

    # Unknown keys are intentionally ignored. In particular, the old `preset`
    # key changed model precision/cache/simplification and is not carried into
    # the v2 safety-preset UI. The concrete values it previously selected are
    # already stored separately and remain untouched.
    for key, value in saved_mapping.items():
        if key in normalized:
            normalized[key] = value

    normalized.pop('preset', None)
    normalized['frame_rate_limit'] = str(normalized['frame_rate_limit'])
    normalized['gpu_duty_limit'] = str(normalized['gpu_duty_limit'])

    if normalized.get('ram_cache_mode') not in ('raw', 'brotli'):
        normalized['ram_cache_mode'] = 'raw'

    actual_preset = infer_safety_preset(
        normalized['frame_rate_limit'],
        normalized['gpu_duty_limit'],
    )
    saved_preset = saved_mapping.get('safety_preset')
    if saved_preset == 'Custom' and actual_preset == 'Custom':
        normalized['safety_preset'] = 'Custom'
    else:
        # The values are authoritative. This also migrates legacy configs and
        # repairs hand-edited files whose preset label disagrees with the pair.
        normalized['safety_preset'] = actual_preset

    normalized['config_version'] = LAUNCHER_CONFIG_VERSION
    return normalized


def load_launcher_config(path='launcher.json', defaults=None):
    """Read a launcher config, falling back to validated defaults on failure."""
    try:
        with open(path, encoding='utf-8') as config_file:
            saved = json.load(config_file)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        saved = {}
    return normalize_launcher_config(saved, defaults=defaults)


def save_launcher_config(config, path='launcher.json'):
    """Atomically save a normalized v2 launcher config."""
    normalized = normalize_launcher_config(config)
    destination = Path(path)
    temporary = destination.with_name(destination.name + '.tmp')
    try:
        with open(temporary, 'w', encoding='utf-8', newline='\n') as config_file:
            json.dump(normalized, config_file, ensure_ascii=False, indent=2)
            config_file.write('\n')
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return normalized


def parse_cache_size_bytes(size):
    match = re.fullmatch(r'(\d+(?:\.\d+)?)(b|kb|mb|gb|tb)', str(size).lower())
    if match is None:
        raise ValueError(f'Invalid cache size: {size}')
    amount = float(match.group(1))
    unit_index = ('b', 'kb', 'mb', 'gb', 'tb').index(match.group(2))
    return int(amount * (1024 ** unit_index))


def describe_ram_cache(
    size,
    storage_mode,
    super_resolution=False,
    simplify_enabled=True,
):
    """Create a concise, explicitly approximate launcher cache explanation."""
    size_bytes = parse_cache_size_bytes(size)
    if size_bytes <= 0:
        return '已关闭；不会占用额外的最终帧 RAM 缓存。'
    if not simplify_enabled:
        return '输入简化已关闭，因此最终帧 RAM 缓存不会启用。'

    size_gib = size_bytes / (1024 ** 3)
    size_label = f'{size_gib:g} GiB'
    if storage_mode == 'brotli':
        sr_note = '；开启超分时仍按 1:4 共享预算' if super_resolution else ''
        return (
            f'{size_label} 总预算；Brotli 节省内存但命中时需要解压'
            f'{sr_note}，实际容量取决于画面内容。'
        )
    if storage_mode != 'raw':
        raise ValueError(f'Unknown RAM cache storage mode: {storage_mode}')

    frame_512_bytes = 512 * 512 * 4
    if super_resolution:
        # Base/SR frames use 1 MiB/4 MiB, so the 1:4 budget split stores an
        # approximately equal count on both sides of the pipeline.
        frame_pairs = math.floor((size_bytes / 5) / frame_512_bytes)
        return (
            f'{size_label} raw 总预算；按 1:4 分配后约各容纳 '
            f'{frame_pairs} 张 512²/1024² BGRA 帧。实际容量会略低。'
        )

    frames = math.floor(size_bytes / frame_512_bytes)
    return (
        f'{size_label} raw 约容纳 {frames} 张 512² BGRA 帧；'
        '低延迟，但长会话会使用更多内存。'
    )


def build_launch_command(
    config,
    python_executable,
    display_size,
    preview_shm_name=None,
):
    """Build the exact argv passed to src.main without depending on wx."""
    settings = normalize_launcher_config(config)
    run_args = [str(python_executable), '-m', 'src.main']

    character = settings['character'] or ''
    if character:
        run_args.extend(('--character', str(character)))

    input_mode = settings['input']
    if input_mode == 0:
        ifm_address = settings['ifm'] or ''
        if ifm_address:
            if ':' not in ifm_address:
                ifm_address += ':49983'
            run_args.extend(('--ifm_input', ifm_address))
    elif input_mode == 1:
        run_args.append('--cam_input')
    elif input_mode == 2:
        run_args.append('--debug_input')
    elif input_mode == 3:
        width, height = display_size
        run_args.extend(('--mouse_input', f'0,0,{width},{height}'))
        if settings['mouse_audio_input']:
            run_args.append('--mouse_audio_input')
            if settings['audio_sensitivity']:
                run_args.extend(
                    ('--audio_sensitivity', str(settings['audio_sensitivity']))
                )
            if settings['audio_threshold']:
                run_args.extend(
                    ('--audio_threshold', str(settings['audio_threshold']))
                )
        if settings['blink_interval']:
            run_args.extend(('--blink_interval', str(settings['blink_interval'])))
    elif input_mode == 4:
        osf_address = settings['osf'] or ''
        if osf_address:
            run_args.extend(('--osf_input', osf_address))

    if settings['breath_cycle']:
        run_args.extend(('--breath_cycle', str(settings['breath_cycle'])))

    output_mode = settings['output']
    if output_mode == 0:
        run_args.append('--output_spout2')
    elif output_mode == 1:
        run_args.append('--output_virtual_cam')
    elif output_mode == 2:
        run_args.append('--output_debug')

    boolean_flags = (
        ('is_alpha_split', '--alpha_split'),
        ('is_extend_movement', '--extend_movement'),
        ('is_bongo', '--bongo'),
        ('is_alpha_clean', '--alpha_clean'),
        ('is_eyebrow', '--eyebrow'),
    )
    for key, flag in boolean_flags:
        if settings[key]:
            run_args.append(flag)

    simplify_name = settings['cache_simplify']
    if simplify_name is not None:
        run_args.extend(('--simplify', str(CACHE_SIMPLIFY_MAP[simplify_name])))

    if settings['ram_cache_size'] is not None:
        run_args.extend(('--cache', settings['ram_cache_size']))
        run_args.extend(('--ram_cache_mode', settings['ram_cache_mode']))
        run_args.extend(('--gpu_cache', settings['vram_cache_size']))

    interpolation = settings['interpolation']
    if interpolation is not None:
        if interpolation != 'Off':
            run_args.append('--use_interpolation')
        if 'half' in interpolation:
            run_args.append('--interpolation_half')
        for scale in (2, 3, 4):
            if f'x{scale}' in interpolation:
                run_args.extend(('--interpolation_scale', str(scale)))
                break

    model_select = settings['model_select']
    if model_select is not None:
        if 'tha4_student_' in model_select:
            run_args.extend(('--model_version', 'v4_student'))
            run_args.extend(
                ('--model_name', model_select.replace('tha4_student_', ''))
            )
        elif 'tha4' in model_select:
            run_args.extend(('--model_version', 'v4'))
        else:
            run_args.extend(('--model_version', 'v3'))
        if 'seperable' in model_select:
            run_args.append('--model_seperable')
        if 'half' in model_select:
            run_args.append('--model_half')

    if settings['frame_rate_limit'] is not None:
        run_args.extend(('--frame_rate_limit', settings['frame_rate_limit']))
    if settings['gpu_duty_limit'] is not None:
        run_args.extend(('--gpu_duty_limit', settings['gpu_duty_limit']))

    if not settings['use_tensorrt'] and settings['dml_device'] != 'auto':
        run_args.extend(('--dml_device_id', settings['dml_device']))

    super_resolution = settings['sr']
    if super_resolution is not None and super_resolution != 'Off':
        run_args.append('--use_sr')
        if 'anime4k' in super_resolution:
            run_args.append('--sr_a4k')
        if 'x4' in super_resolution:
            run_args.append('--sr_x4')
        if 'half' in super_resolution:
            run_args.append('--sr_half')

    if settings['use_tensorrt']:
        run_args.append('--use_tensorrt')

    run_args.extend(
        ('--filter_min_cutoff', str(min_cutoff_mapper(settings['min_cutoff'])))
    )
    run_args.extend(('--filter_beta', str(beta_mapper(settings['beta']))))
    if preview_shm_name:
        run_args.extend(('--preview_shm', str(preview_shm_name)))
    return run_args
