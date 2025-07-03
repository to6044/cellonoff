#!/usr/bin/env python3
"""
EXP3 Cell On/Off 결과 분석 스크립트
사용법: python analyze_exp3_results.py --results_dir data/output/exp3_cell_onoff_YYYYMMDD_HHMMSS
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_exp3_results(results_dir):
    """EXP3 결과 파일 로드"""
    results_path = Path(results_dir)
    
    # Load JSON results
    with open(results_path / 'exp3_weights_history.json', 'r') as f:
        results = json.load(f)
    
    # Load TSV simulation data if exists
    tsv_files = list(results_path.glob('*.tsv'))
    sim_data = None
    if tsv_files:
        sim_data = pd.read_csv(tsv_files[0], sep='\t')
    
    return results, sim_data

def analyze_convergence(results):
    """수렴 분석"""
    probs_history = np.array(results['probabilities_history'])
    
    # Calculate entropy over time
    entropy = -np.sum(probs_history * np.log(probs_history + 1e-10), axis=1)
    
    # Find convergence point (when entropy stabilizes)
    entropy_diff = np.diff(entropy)
    convergence_idx = None
    
    for i in range(len(entropy_diff) - 10):
        if np.std(entropy_diff[i:i+10]) < 0.01:
            convergence_idx = i
            break
    
    return entropy, convergence_idx

def analyze_arm_performance(results):
    """각 arm의 성능 분석"""
    arms = results['arms']
    rewards = results['rewards']
    arm_selections = results['arm_selections']
    
    # Calculate average reward per arm
    arm_rewards = {}
    for i, arm_idx in enumerate(arm_selections):
        if arm_idx not in arm_rewards:
            arm_rewards[arm_idx] = []
        arm_rewards[arm_idx].append(rewards[i])
    
    arm_stats = []
    for arm_idx in range(len(arms)):
        if arm_idx in arm_rewards:
            arm_stats.append({
                'arm_idx': arm_idx,
                'cells_off': arms[arm_idx],
                'avg_reward': np.mean(arm_rewards[arm_idx]),
                'std_reward': np.std(arm_rewards[arm_idx]),
                'count': len(arm_rewards[arm_idx]),
                'final_prob': results['probabilities_history'][-1][arm_idx]
            })
    
    return pd.DataFrame(arm_stats)

def create_detailed_analysis_plots(results, sim_data, output_dir):
    """상세 분석 플롯 생성"""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Entropy over time (수렴 분석)
    ax1 = plt.subplot(3, 3, 1)
    entropy, conv_idx = analyze_convergence(results)
    ax1.plot(entropy)
    if conv_idx:
        ax1.axvline(x=conv_idx, color='r', linestyle='--', label=f'Convergence at {conv_idx}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Entropy')
    ax1.set_title('Selection Entropy Over Time')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Efficiency improvement
    ax2 = plt.subplot(3, 3, 2)
    efficiency = [e[0] for e in results['efficiency_history']]
    window_size = 20
    moving_avg = pd.Series(efficiency).rolling(window_size).mean()
    ax2.plot(efficiency, alpha=0.3, label='Raw')
    ax2.plot(moving_avg, label=f'{window_size}-episode MA')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Network Efficiency (Mbps/kW)')
    ax2.set_title('Efficiency Improvement')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Throughput vs Power trade-off
    ax3 = plt.subplot(3, 3, 3)
    throughputs = [e[1] for e in results['efficiency_history']]
    powers = [e[2] for e in results['efficiency_history']]
    scatter = ax3.scatter(powers, throughputs, c=range(len(powers)), 
                         cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, ax=ax3, label='Episode')
    ax3.set_xlabel('Total Power (kW)')
    ax3.set_ylabel('Total Throughput (Mbps)')
    ax3.set_title('Throughput vs Power Trade-off')
    ax3.grid(True)
    
    # 4. Top arms performance
    ax4 = plt.subplot(3, 3, 4)
    arm_stats = analyze_arm_performance(results)
    top_arms = arm_stats.nlargest(10, 'final_prob')
    ax4.barh(range(len(top_arms)), top_arms['avg_reward'])
    ax4.set_yticks(range(len(top_arms)))
    ax4.set_yticklabels([f"Arm {row['arm_idx']}" for _, row in top_arms.iterrows()])
    ax4.set_xlabel('Average Reward')
    ax4.set_title('Top 10 Arms by Final Probability')
    ax4.grid(True, axis='x')
    
    # 5. Learning curve comparison
    ax5 = plt.subplot(3, 3, 5)
    warm_up_episodes = results.get('warm_up_episodes', 100)
    
    pre_warmup = efficiency[:warm_up_episodes] if len(efficiency) > warm_up_episodes else efficiency
    post_warmup = efficiency[warm_up_episodes:] if len(efficiency) > warm_up_episodes else []
    
    if pre_warmup:
        ax5.plot(range(len(pre_warmup)), pre_warmup, label='Warm-up', alpha=0.7)
    if post_warmup:
        ax5.plot(range(warm_up_episodes, warm_up_episodes + len(post_warmup)), 
                post_warmup, label='EXP3 Learning', alpha=0.7)
    
    ax5.axvline(x=warm_up_episodes, color='r', linestyle='--', label='Start EXP3')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Efficiency (Mbps/kW)')
    ax5.set_title('Learning Phases Comparison')
    ax5.legend()
    ax5.grid(True)
    
    # 6. Cell selection frequency heatmap
    ax6 = plt.subplot(3, 3, 6)
    n_cells = max(max(arm) for arm in results['arms']) + 1
    cell_off_freq = np.zeros(n_cells)
    
    for i, arm_idx in enumerate(results['arm_selections']):
        for cell_idx in results['arms'][arm_idx]:
            cell_off_freq[cell_idx] += 1
    
    cell_off_freq = cell_off_freq / len(results['arm_selections'])
    
    ax6.bar(range(n_cells), cell_off_freq)
    ax6.set_xlabel('Cell Index')
    ax6.set_ylabel('Fraction of Time OFF')
    ax6.set_title('Cell OFF Frequency')
    ax6.grid(True, axis='y')
    
    # 7. Reward distribution by warm-up phase
    ax7 = plt.subplot(3, 3, 7)
    if len(results['rewards']) > warm_up_episodes:
        ax7.hist(results['rewards'][:warm_up_episodes], bins=30, alpha=0.5, 
                label='Warm-up', density=True)
        ax7.hist(results['rewards'][warm_up_episodes:], bins=30, alpha=0.5, 
                label='EXP3', density=True)
        ax7.set_xlabel('Reward')
        ax7.set_ylabel('Density')
        ax7.set_title('Reward Distribution by Phase')
        ax7.legend()
        ax7.grid(True, axis='y')
    
    # 8. Best configuration details
    ax8 = plt.subplot(3, 3, 8)
    best_arm_idx = np.argmax(results['probabilities_history'][-1])
    best_arm = results['arms'][best_arm_idx]
    
    # Create cell grid visualization
    cells_on = set(range(n_cells)) - set(best_arm)
    cell_status = ['OFF' if i in best_arm else 'ON' for i in range(n_cells)]
    colors = ['red' if s == 'OFF' else 'green' for s in cell_status]
    
    ax8.bar(range(n_cells), [1]*n_cells, color=colors, edgecolor='black')
    ax8.set_xlabel('Cell Index')
    ax8.set_title(f'Best Configuration (Prob: {results["probabilities_history"][-1][best_arm_idx]:.3f})')
    ax8.set_ylim(0, 1.5)
    ax8.set_yticks([])
    
    # Add text
    ax8.text(0.5, 0.5, f'Cells OFF: {sorted(best_arm)}\nAvg Efficiency: {np.mean(efficiency[-20:]):.2f} Mbps/kW',
            transform=ax8.transAxes, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 9. Probability evolution for top 5 arms
    ax9 = plt.subplot(3, 3, 9)
    probs_array = np.array(results['probabilities_history'])
    final_probs = probs_array[-1]
    top_5_arms = np.argsort(final_probs)[-5:][::-1]
    
    for arm_idx in top_5_arms:
        ax9.plot(probs_array[:, arm_idx], 
                label=f'Arm {arm_idx} (off: {len(results["arms"][arm_idx])} cells)')
    
    ax9.set_xlabel('Episode')
    ax9.set_ylabel('Selection Probability')
    ax9.set_title('Top 5 Arms Probability Evolution')
    ax9.legend(fontsize=8)
    ax9.grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'exp3_detailed_analysis.png', dpi=300)
    plt.close()

def generate_summary_report(results, sim_data, output_dir):
    """요약 보고서 생성"""
    report = []
    report.append("=== EXP3 Cell On/Off Analysis Report ===\n")
    
    # Basic statistics
    report.append(f"Total episodes: {len(results['arm_selections'])}")
    report.append(f"Number of arms: {len(results['arms'])}")
    report.append(f"Cells per arm turned off: {len(results['arms'][0])}")
    
    # Convergence analysis
    entropy, conv_idx = analyze_convergence(results)
    if conv_idx:
        report.append(f"\nConvergence detected at episode: {conv_idx}")
    
    # Performance improvement
    efficiency = [e[0] for e in results['efficiency_history']]
    initial_avg = np.mean(efficiency[:20]) if len(efficiency) > 20 else np.mean(efficiency)
    final_avg = np.mean(efficiency[-20:]) if len(efficiency) > 20 else np.mean(efficiency)
    improvement = ((final_avg - initial_avg) / initial_avg) * 100
    
    report.append(f"\nPerformance Improvement:")
    report.append(f"  Initial efficiency: {initial_avg:.2f} Mbps/kW")
    report.append(f"  Final efficiency: {final_avg:.2f} Mbps/kW")
    report.append(f"  Improvement: {improvement:.1f}%")
    
    # Best configuration
    best_arm_idx = np.argmax(results['probabilities_history'][-1])
    best_arm = results['arms'][best_arm_idx]
    best_prob = results['probabilities_history'][-1][best_arm_idx]
    
    report.append(f"\nBest Configuration:")
    report.append(f"  Arm index: {best_arm_idx}")
    report.append(f"  Cells to turn off: {sorted(best_arm)}")
    report.append(f"  Selection probability: {best_prob:.3f}")
    
    # Top 5 arms
    arm_stats = analyze_arm_performance(results)
    top_arms = arm_stats.nlargest(5, 'final_prob')
    
    report.append(f"\nTop 5 Arms by Final Probability:")
    for _, row in top_arms.iterrows():
        report.append(f"  Arm {row['arm_idx']}: prob={row['final_prob']:.3f}, "
                     f"avg_reward={row['avg_reward']:.2f}, count={row['count']}")
    
    # Save report
    with open(Path(output_dir) / 'exp3_analysis_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print('\n'.join(report))

def main():
    parser = argparse.ArgumentParser(description='Analyze EXP3 Cell On/Off results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to results directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis (default: same as results_dir)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    results, sim_data = load_exp3_results(args.results_dir)
    
    # Generate analysis
    print("Generating detailed analysis plots...")
    create_detailed_analysis_plots(results, sim_data, args.output_dir)
    
    print("Generating summary report...")
    generate_summary_report(results, sim_data, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
