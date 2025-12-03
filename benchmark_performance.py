#!/usr/bin/env python3
"""
性能基准测试脚本
用于对比优化前后的训练性能
"""

import os
import sys
import time
import json
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple


class Color:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{Color.HEADER}{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}{Color.ENDC}\n")


def run_benchmark(config_path: str, steps: int = 10) -> Dict:
    """运行基准测试"""
    print(f"\n{Color.OKBLUE}测试配置: {config_path}{Color.ENDC}")
    print(f"{Color.OKBLUE}测试步数: {steps}{Color.ENDC}\n")

    cmd = [
        sys.executable, 'train_model.py',
        '--config', config_path,
        '--max_steps', str(steps)
    ]

    print(f"运行命令: {' '.join(cmd)}\n")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()

    duration = end_time - start_time
    success = result.returncode == 0

    # 解析输出中的性能信息
    output = result.stdout + result.stderr

    benchmark_result = {
        'config': config_path,
        'steps': steps,
        'duration_seconds': duration,
        'success': success,
        'throughput_steps_per_second': steps / duration if success else 0,
        'avg_seconds_per_step': duration / steps if success else 0,
        'timestamp': datetime.now().isoformat()
    }

    if success:
        print(f"{Color.OKGREEN}✓ 测试完成{Color.ENDC}")
        print(f"  总耗时: {duration:.2f} 秒")
        print(f"  平均每步: {duration/steps:.2f} 秒")
        print(f"  吞吐量: {steps/duration:.2f} steps/秒")
    else:
        print(f"{Color.FAIL}✗ 测试失败{Color.ENDC}")
        print(f"  错误输出: {result.stderr[:500]}")

    return benchmark_result


def check_gpu_stats() -> Dict:
    """检查 GPU 状态"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw,power.limit',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )

        parts = result.stdout.strip().split(', ')

        return {
            'name': parts[0],
            'memory_total_mb': float(parts[1]),
            'memory_used_mb': float(parts[2]),
            'gpu_util_percent': float(parts[3]),
            'power_draw_w': float(parts[4]),
            'power_limit_w': float(parts[5])
        }
    except Exception as e:
        print(f"{Color.WARNING}⚠ 无法获取 GPU 信息: {e}{Color.ENDC}")
        return {}


def compare_results(baseline: Dict, optimized: Dict):
    """对比基准和优化结果"""
    print_section("性能对比")

    if not baseline['success'] or not optimized['success']:
        print(f"{Color.FAIL}无法对比：部分测试失败{Color.ENDC}")
        return

    # 计算改进比例
    speedup = baseline['duration_seconds'] / optimized['duration_seconds']
    throughput_improvement = (
        (optimized['throughput_steps_per_second'] - baseline['throughput_steps_per_second']) /
        baseline['throughput_steps_per_second'] * 100
    )

    print(f"{Color.BOLD}训练速度:{Color.ENDC}")
    print(f"  基准配置: {baseline['avg_seconds_per_step']:.3f} 秒/步")
    print(f"  优化配置: {optimized['avg_seconds_per_step']:.3f} 秒/步")
    print(f"  {Color.OKGREEN}加速比: {speedup:.2f}x{Color.ENDC}")

    print(f"\n{Color.BOLD}吞吐量:{Color.ENDC}")
    print(f"  基准配置: {baseline['throughput_steps_per_second']:.3f} steps/秒")
    print(f"  优化配置: {optimized['throughput_steps_per_second']:.3f} steps/秒")
    print(f"  {Color.OKGREEN}提升: {throughput_improvement:+.1f}%{Color.ENDC}")

    print(f"\n{Color.BOLD}总耗时 ({baseline['steps']} 步):{Color.ENDC}")
    print(f"  基准配置: {baseline['duration_seconds']:.2f} 秒")
    print(f"  优化配置: {optimized['duration_seconds']:.2f} 秒")
    print(f"  {Color.OKGREEN}节省: {baseline['duration_seconds'] - optimized['duration_seconds']:.2f} 秒{Color.ENDC}")

    # 性能等级评估
    print(f"\n{Color.BOLD}性能评估:{Color.ENDC}")
    if speedup >= 2.5:
        print(f"  {Color.OKGREEN}★★★★★ 优秀 - 加速比达到 {speedup:.1f}x{Color.ENDC}")
    elif speedup >= 2.0:
        print(f"  {Color.OKGREEN}★★★★☆ 良好 - 加速比达到 {speedup:.1f}x{Color.ENDC}")
    elif speedup >= 1.5:
        print(f"  {Color.OKCYAN}★★★☆☆ 中等 - 加速比达到 {speedup:.1f}x{Color.ENDC}")
    elif speedup >= 1.2:
        print(f"  {Color.WARNING}★★☆☆☆ 一般 - 加速比仅 {speedup:.1f}x{Color.ENDC}")
    else:
        print(f"  {Color.FAIL}★☆☆☆☆ 较差 - 加速比仅 {speedup:.1f}x{Color.ENDC}")
        print(f"  建议检查配置或硬件状态")


def save_results(baseline: Dict, optimized: Dict, output_file: str):
    """保存测试结果"""
    results = {
        'test_time': datetime.now().isoformat(),
        'baseline': baseline,
        'optimized': optimized,
        'gpu_info': check_gpu_stats()
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{Color.OKGREEN}结果已保存: {output_file}{Color.ENDC}")


def main():
    parser = argparse.ArgumentParser(
        description="性能基准测试 - 对比优化前后的训练速度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本测试（10步）
  python benchmark_performance.py

  # 更长的测试（50步）
  python benchmark_performance.py --steps 50

  # 只测试优化配置
  python benchmark_performance.py --only-optimized

  # 自定义配置对比
  python benchmark_performance.py \\
    --baseline-config my_config.yaml \\
    --optimized-config my_optimized_config.yaml
        """
    )

    parser.add_argument('--baseline-config', type=str, default='train_config.yaml',
                        help='基准配置文件 (默认: train_config.yaml)')
    parser.add_argument('--optimized-config', type=str, default='train_config_optimized.yaml',
                        help='优化配置文件 (默认: train_config_optimized.yaml)')
    parser.add_argument('--steps', type=int, default=10,
                        help='测试步数 (默认: 10)')
    parser.add_argument('--only-baseline', action='store_true',
                        help='只测试基准配置')
    parser.add_argument('--only-optimized', action='store_true',
                        help='只测试优化配置')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='结果保存文件 (默认: benchmark_results.json)')
    parser.add_argument('--skip-comparison', action='store_true',
                        help='跳过对比（只运行测试）')

    args = parser.parse_args()

    print_section("🚀 DeepSeek OCR 性能基准测试")
    print(f"{Color.OKBLUE}开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Color.ENDC}")

    # 检查 GPU
    gpu_info = check_gpu_stats()
    if gpu_info:
        print(f"\n{Color.BOLD}GPU 信息:{Color.ENDC}")
        print(f"  型号: {gpu_info.get('name', 'Unknown')}")
        print(f"  显存: {gpu_info.get('memory_used_mb', 0):.0f}MB / "
              f"{gpu_info.get('memory_total_mb', 0):.0f}MB")
        print(f"  利用率: {gpu_info.get('gpu_util_percent', 0):.0f}%")
        print(f"  功耗: {gpu_info.get('power_draw_w', 0):.0f}W / "
              f"{gpu_info.get('power_limit_w', 0):.0f}W")

    baseline_result = None
    optimized_result = None

    try:
        # 测试基准配置
        if not args.only_optimized:
            print_section("📊 测试基准配置")
            if not os.path.exists(args.baseline_config):
                print(f"{Color.FAIL}错误: 基准配置文件不存在 {args.baseline_config}{Color.ENDC}")
                sys.exit(1)
            baseline_result = run_benchmark(args.baseline_config, args.steps)

        # 测试优化配置
        if not args.only_baseline:
            print_section("📊 测试优化配置")
            if not os.path.exists(args.optimized_config):
                print(f"{Color.FAIL}错误: 优化配置文件不存在 {args.optimized_config}{Color.ENDC}")
                sys.exit(1)
            optimized_result = run_benchmark(args.optimized_config, args.steps)

        # 对比结果
        if baseline_result and optimized_result and not args.skip_comparison:
            compare_results(baseline_result, optimized_result)

        # 保存结果
        if baseline_result or optimized_result:
            save_results(
                baseline_result or {},
                optimized_result or {},
                args.output
            )

        # 总结
        print_section("✅ 测试完成")
        print(f"{Color.OKGREEN}所有测试已完成{Color.ENDC}\n")

        if baseline_result and optimized_result:
            speedup = baseline_result['duration_seconds'] / optimized_result['duration_seconds']
            print(f"{Color.BOLD}关键指标:{Color.ENDC}")
            print(f"  🚀 加速比: {Color.OKGREEN}{speedup:.2f}x{Color.ENDC}")

            if speedup >= 2.0:
                print(f"\n{Color.OKGREEN}✨ 优化效果显著！建议使用优化配置进行训练。{Color.ENDC}")
            elif speedup >= 1.5:
                print(f"\n{Color.OKCYAN}📈 优化有一定效果，可考虑进一步调优。{Color.ENDC}")
            else:
                print(f"\n{Color.WARNING}⚠️  优化效果不明显，建议检查配置或系统状态。{Color.ENDC}")

    except KeyboardInterrupt:
        print(f"\n\n{Color.WARNING}测试被用户中断{Color.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Color.FAIL}测试失败: {e}{Color.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
