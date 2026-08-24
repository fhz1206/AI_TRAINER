"""
resource_limits.py — 平台资源占用上限（默认为一半系统资源）

职责：
1. 配置：CPU 线程数上限、内存占用上限（MB），持久化在 instance/resource_limits.json；
   未配置时默认各取系统资源的一半。
2. 应用：启动时设置 PyTorch/OMP 线程数；DataLoader 工作进程数按上限折算。
3. 准入：新训练启动前检查当前进程内存是否已超上限，超限则拒绝并提示，
   防止多任务叠加把宿主机拖垮。

配置项：
{
  "max_cpu_threads": 8,      # 0 = 不限制（仍不超过物理核数）
  "max_memory_mb": 16384     # 0 = 不限制
}
"""
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, 'instance')
CONFIG_PATH = os.path.join(INSTANCE_DIR, 'resource_limits.json')


def _system_totals():
    """读取系统 CPU 逻辑核数与总内存(MB)；psutil 缺失时尽力降级"""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return os.cpu_count() or 2, int(vm.total / (1024 * 1024))
    except Exception:
        return os.cpu_count() or 2, None


def _defaults():
    cpu_total, mem_total_mb = _system_totals()
    return {
        'max_cpu_threads': max(1, cpu_total // 2),
        'max_memory_mb': (mem_total_mb // 2) if mem_total_mb else 0,
    }


def get_limits():
    """读取生效的资源上限（缺省字段自动补默认值）"""
    cfg = {}
    try:
        with open(CONFIG_PATH, encoding='utf-8') as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        pass
    d = _defaults()
    return {
        'max_cpu_threads': int(cfg.get('max_cpu_threads', d['max_cpu_threads']) or 0),
        'max_memory_mb': int(cfg.get('max_memory_mb', d['max_memory_mb']) or 0),
        'system_cpu_threads': _system_totals()[0],
        'system_memory_mb': _system_totals()[1],
        'is_default': not cfg,
    }


def set_limits(max_cpu_threads=None, max_memory_mb=None):
    """更新并持久化上限；传 0 表示不限制。返回更新后的配置。"""
    cur = get_limits()
    if max_cpu_threads is not None:
        cur['max_cpu_threads'] = max(0, int(max_cpu_threads))
    if max_memory_mb is not None:
        cur['max_memory_mb'] = max(0, int(max_memory_mb))
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump({'max_cpu_threads': cur['max_cpu_threads'],
                   'max_memory_mb': cur['max_memory_mb']},
                  f, ensure_ascii=False, indent=2)
    # 立即生效到计算线程层
    apply_thread_limits()
    return get_limits()


def effective_cpu_threads():
    """DataLoader 等并发场景应使用的线程/进程预算（受上限约束）"""
    total = _system_totals()[0]
    cap = get_limits()['max_cpu_threads']
    return min(total, cap) if cap > 0 else total


def apply_thread_limits():
    """把 CPU 上限应用到当前进程的计算线程池（torch/OMP）"""
    n = effective_cpu_threads()
    try:
        import torch
        torch.set_num_threads(max(1, n))
    except Exception:
        pass
    os.environ.setdefault('OMP_NUM_THREADS', str(n))
    os.environ.setdefault('MKL_NUM_THREADS', str(n))
    return n


def current_usage():
    """当前进程的资源占用快照（供管理端展示与准入判断）"""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss_mb = int(proc.memory_info().rss / (1024 * 1024))
        cpu_pct = proc.cpu_percent(interval=0.1)
        return {'memory_mb': rss_mb, 'cpu_percent': round(cpu_pct, 1)}
    except Exception:
        return {'memory_mb': None, 'cpu_percent': None}


def admission_check():
    """
    训练准入检查：内存超过上限时拒绝新任务。
    返回 (允许?, 原因说明)。
    """
    lim = get_limits()
    usage = current_usage()
    used = usage.get('memory_mb')
    if lim['max_memory_mb'] > 0 and used is not None and used >= lim['max_memory_mb']:
        return False, (f"当前内存占用 {used} MB 已达上限 {lim['max_memory_mb']} MB，"
                       f"请等待运行中的训练结束或调高上限后再启动")
    return True, ''


# ==================== 后台监控 ====================
_monitor_started = False


def start_limit_monitor(interval_seconds=30):
    """
    启动后台守护线程：持续检测本进程资源占用，接近/达到上限时打印告警。
    仅告警不杀进程——训练准入检查已阻止超限后的新任务。
    """
    global _monitor_started
    if _monitor_started:
        return
    _monitor_started = True

    import threading
    import time as _time

    def _loop():
        while True:
            try:
                lim = get_limits()
                usage = current_usage()
                used = usage.get('memory_mb')
                if used is not None:
                    if lim['max_memory_mb'] > 0 and used >= lim['max_memory_mb']:
                        print(f"[ResourceLimit] ⚠️ 内存 {used} MB ≥ 上限 "
                              f"{lim['max_memory_mb']} MB，新训练将被拒绝")
                    elif (lim['max_memory_mb'] > 0
                          and used >= lim['max_memory_mb'] * 0.85):
                        print(f"[ResourceLimit] 内存 {used} MB 接近上限 "
                              f"{lim['max_memory_mb']} MB")
            except Exception:
                pass
            _time.sleep(interval_seconds)

    threading.Thread(target=_loop, daemon=True,
                     name='resource-limit-monitor').start()
