"""
Generate Publication-Ready Figures from GAP-4 Results
Creates high-quality, publication-ready visualizations for papers/presentations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from scipy import stats

# Set publication style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16


def load_results(results_dir):
    """Load GAP-4 results"""
    results_path = Path(results_dir)
    
    # Load CSV
    csv_path = results_path / 'asm_metrics.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Results not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Load statistics
    stats_path = results_path / 'statistics.json'
    with open(stats_path, 'r') as f:
        stats_data = json.load(f)
    
    return df, stats_data


def figure_1_asm_overview(df, save_path):
    """
    Figure 1: ASM Overview
    Multi-panel figure showing ASM distributions and trends
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel A: ASM Distribution (Histogram)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['asm_score'], bins=25, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(df['asm_score'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {df["asm_score"].mean():.4f}')
    ax1.set_xlabel('Attention Shift Metric (ASM)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('(A) ASM Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel B: ASM vs Attack Success
    ax2 = fig.add_subplot(gs[0, 1])
    success_data = [
        df[df['attack_success'] == 0]['asm_score'],
        df[df['attack_success'] == 1]['asm_score']
    ]
    bp = ax2.boxplot(success_data, labels=['Failed', 'Success'], 
                     patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], ['lightcoral', 'lightgreen']):
        patch.set_facecolor(color)
    ax2.set_ylabel('Attention Shift Metric (ASM)')
    ax2.set_title('(B) ASM by Attack Outcome', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistical test
    failed_asm = df[df['attack_success'] == 0]['asm_score']
    success_asm = df[df['attack_success'] == 1]['asm_score']
    if len(failed_asm) > 0 and len(success_asm) > 0:
        t_stat, p_val = stats.ttest_ind(failed_asm, success_asm)
        ax2.text(0.5, 0.95, f'p-value: {p_val:.4f}', 
                transform=ax2.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel C: ASM vs Epsilon
    ax3 = fig.add_subplot(gs[0, 2])
    epsilon_groups = df.groupby('epsilon')['asm_score'].agg(['mean', 'std', 'count'])
    epsilon_values = epsilon_groups.index.values
    means = epsilon_groups['mean'].values
    stds = epsilon_groups['std'].values
    
    ax3.errorbar(epsilon_values * 255, means, yerr=stds, 
                marker='o', markersize=8, linewidth=2, capsize=5, 
                color='steelblue', label='Mean ± SD')
    ax3.set_xlabel('Attack Strength (eps / 255)')
    ax3.set_ylabel('Mean ASM')
    ax3.set_title('(C) ASM vs Attack Strength', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Panel D: ASM Heatmap by Attack and Epsilon
    ax4 = fig.add_subplot(gs[1, 0])
    pivot = df.pivot_table(values='asm_score', index='attack', columns='epsilon', aggfunc='mean')
    im = ax4.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(len(pivot.columns)))
    ax4.set_xticklabels([f'{e*255:.1f}' for e in pivot.columns])
    ax4.set_yticks(range(len(pivot.index)))
    ax4.set_yticklabels([a.upper() for a in pivot.index])
    ax4.set_xlabel('Attack Strength (eps / 255)')
    ax4.set_ylabel('Attack Type')
    ax4.set_title('(D) Mean ASM Heatmap', fontweight='bold')
    
    # Add values to heatmap
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax4.text(j, i, f'{pivot.values[i, j]:.3f}', 
                    ha='center', va='center', color='black', fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Mean ASM')
    
    # Panel E: Attack Success Rate vs Epsilon
    ax5 = fig.add_subplot(gs[1, 1])
    asr_by_eps = df.groupby('epsilon')['attack_success'].mean() * 100
    ax5.plot(asr_by_eps.index * 255, asr_by_eps.values, 
            marker='s', markersize=8, linewidth=2, color='darkred')
    ax5.set_xlabel('Attack Strength (eps / 255)')
    ax5.set_ylabel('Attack Success Rate (%)')
    ax5.set_title('(E) ASR vs Attack Strength', fontweight='bold')
    ax5.set_ylim([0, 105])
    ax5.grid(True, alpha=0.3)
    
    # Panel F: Correlation Plot
    ax6 = fig.add_subplot(gs[1, 2])
    scatter = ax6.scatter(df['asm_score'], df['attack_success'], 
                         alpha=0.4, s=50, c=df['epsilon']*255, 
                         cmap='viridis', edgecolors='black', linewidth=0.5)
    ax6.set_xlabel('Attention Shift Metric (ASM)')
    ax6.set_ylabel('Attack Success (0=Failed, 1=Success)')
    ax6.set_title('(F) ASM vs Attack Success', fontweight='bold')
    ax6.set_ylim([-0.1, 1.1])
    ax6.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('eps / 255')
    
    # Add correlation line
    if df['asm_score'].std() > 0:
        z = np.polyfit(df['asm_score'], df['attack_success'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['asm_score'].min(), df['asm_score'].max(), 100)
        ax6.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label='Trend')
        ax6.legend()
    
    fig.suptitle('Attention Disruption Analysis: Comprehensive Overview', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 1: {save_path}")
    plt.close()


def figure_2_attack_comparison(df, save_path):
    """
    Figure 2: Attack Type Comparison
    Compare FGSM vs PGD in terms of ASM
    """
    if 'fgsm' not in df['attack'].values or 'pgd' not in df['attack'].values:
        print("Skipping Figure 2: Need both FGSM and PGD data")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Box plot comparison
    attack_types = df['attack'].unique()
    data_by_attack = [df[df['attack'] == att]['asm_score'] for att in attack_types]
    
    bp = axes[0].boxplot(data_by_attack, labels=[a.upper() for a in attack_types],
                         patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)
    
    axes[0].set_ylabel('Attention Shift Metric (ASM)')
    axes[0].set_title('(A) ASM Distribution by Attack', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Panel B: ASR comparison
    asr_by_attack = df.groupby('attack')['attack_success'].mean() * 100
    bars = axes[1].bar(range(len(asr_by_attack)), asr_by_attack.values, 
                       color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
    axes[1].set_xticks(range(len(asr_by_attack)))
    axes[1].set_xticklabels([a.upper() for a in asr_by_attack.index])
    axes[1].set_ylabel('Attack Success Rate (%)')
    axes[1].set_title('(B) ASR by Attack Type', fontweight='bold')
    axes[1].set_ylim([0, 105])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Panel C: Scatter comparison
    for attack in attack_types:
        attack_df = df[df['attack'] == attack]
        axes[2].scatter(attack_df['epsilon'] * 255, attack_df['asm_score'],
                       label=attack.upper(), alpha=0.5, s=30)
    
    axes[2].set_xlabel('Attack Strength (eps / 255)')
    axes[2].set_ylabel('Attention Shift Metric (ASM)')
    axes[2].set_title('(C) ASM vs Epsilon by Attack', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle('Attack Type Comparison: FGSM vs PGD', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 2: {save_path}")
    plt.close()


def figure_3_epsilon_analysis(df, save_path):
    """
    Figure 3: Epsilon Sensitivity Analysis
    Detailed analysis of how attack strength affects ASM
    """
    epsilon_values = sorted(df['epsilon'].unique())
    
    if len(epsilon_values) < 2:
        print("Skipping Figure 3: Need multiple epsilon values")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Box plots by epsilon
    data_by_eps = [df[df['epsilon'] == eps]['asm_score'] for eps in epsilon_values]
    labels = [f'{eps*255:.1f}' for eps in epsilon_values]
    
    bp = axes[0, 0].boxplot(data_by_eps, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    axes[0, 0].set_xlabel('Attack Strength (eps / 255)')
    axes[0, 0].set_ylabel('Attention Shift Metric (ASM)')
    axes[0, 0].set_title('(A) ASM Distribution by Epsilon', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Panel B: Violin plots
    eps_labels = [f'{eps*255:.1f}' for eps in df['epsilon']]
    df_plot = df.copy()
    df_plot['eps_label'] = eps_labels
    
    axes[0, 1].violinplot([df[df['epsilon'] == eps]['asm_score'] for eps in epsilon_values],
                          positions=range(len(epsilon_values)), showmeans=True)
    axes[0, 1].set_xticks(range(len(epsilon_values)))
    axes[0, 1].set_xticklabels([f'{eps*255:.1f}' for eps in epsilon_values])
    axes[0, 1].set_xlabel('Attack Strength (eps / 255)')
    axes[0, 1].set_ylabel('Attention Shift Metric (ASM)')
    axes[0, 1].set_title('(B) ASM Density by Epsilon', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Panel C: Mean and confidence intervals
    stats_by_eps = df.groupby('epsilon')['asm_score'].agg(['mean', 'std', 'count'])
    ci = 1.96 * stats_by_eps['std'] / np.sqrt(stats_by_eps['count'])
    
    x_vals = epsilon_values * 255
    axes[1, 0].plot(x_vals, stats_by_eps['mean'], 'o-', linewidth=2, markersize=8,
                   color='steelblue', label='Mean ASM')
    axes[1, 0].fill_between(x_vals, 
                            stats_by_eps['mean'] - ci, 
                            stats_by_eps['mean'] + ci,
                            alpha=0.3, color='steelblue', label='95% CI')
    axes[1, 0].set_xlabel('Attack Strength (eps / 255)')
    axes[1, 0].set_ylabel('Mean ASM')
    axes[1, 0].set_title('(C) ASM Trend with Confidence Intervals', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel D: Success rate comparison
    asr_by_eps = df.groupby('epsilon')['attack_success'].mean() * 100
    mean_asm_by_eps = df.groupby('epsilon')['asm_score'].mean()
    
    ax_d1 = axes[1, 1]
    ax_d2 = ax_d1.twinx()
    
    line1 = ax_d1.plot(epsilon_values * 255, asr_by_eps.values, 
                       'o-', color='red', linewidth=2, markersize=8, label='ASR')
    line2 = ax_d2.plot(epsilon_values * 255, mean_asm_by_eps.values, 
                       's-', color='blue', linewidth=2, markersize=8, label='Mean ASM')
    
    ax_d1.set_xlabel('Attack Strength (eps / 255)')
    ax_d1.set_ylabel('Attack Success Rate (%)', color='red')
    ax_d2.set_ylabel('Mean ASM', color='blue')
    ax_d1.tick_params(axis='y', labelcolor='red')
    ax_d2.tick_params(axis='y', labelcolor='blue')
    ax_d1.set_title('(D) ASR and ASM vs Epsilon', fontweight='bold')
    ax_d1.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_d1.legend(lines, labels, loc='upper left')
    
    fig.suptitle('Epsilon Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 3: {save_path}")
    plt.close()


def generate_summary_table(df, stats_data, save_path):
    """Generate summary statistics table"""
    # Overall statistics
    overall = {
        'Metric': ['Mean ASM', 'Std ASM', 'Median ASM', 'Min ASM', 'Max ASM', 
                  'ASR (%)', 'Samples'],
        'Value': [
            f"{stats_data['statistics']['mean_asm']:.4f}",
            f"{stats_data['statistics']['std_asm']:.4f}",
            f"{stats_data['statistics']['median_asm']:.4f}",
            f"{stats_data['statistics']['min_asm']:.4f}",
            f"{stats_data['statistics']['max_asm']:.4f}",
            f"{stats_data['statistics']['asr']*100:.2f}",
            f"{stats_data['statistics']['num_samples']}"
        ]
    }
    
    # By epsilon
    by_epsilon = df.groupby('epsilon').agg({
        'asm_score': ['mean', 'std'],
        'attack_success': 'mean',
        'sample_id': 'count'
    }).round(4)
    
    # By attack
    by_attack = df.groupby('attack').agg({
        'asm_score': ['mean', 'std'],
        'attack_success': 'mean',
        'sample_id': 'count'
    }).round(4)
    
    # Create figure with tables
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('Summary Statistics', fontsize=16, fontweight='bold')
    
    # Table 1: Overall
    axes[0].axis('tight')
    axes[0].axis('off')
    table1 = axes[0].table(cellText=[overall['Metric'], overall['Value']],
                           rowLabels=['Metric', 'Value'],
                           loc='center', cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 2)
    axes[0].set_title('(A) Overall Statistics', fontweight='bold', pad=20)
    
    # Table 2: By Epsilon
    axes[1].axis('tight')
    axes[1].axis('off')
    eps_data = []
    eps_data.append(['Epsilon'] + [f'{e*255:.1f}' for e in by_epsilon.index])
    eps_data.append(['Mean ASM'] + [f'{v:.4f}' for v in by_epsilon[('asm_score', 'mean')]])
    eps_data.append(['Std ASM'] + [f'{v:.4f}' for v in by_epsilon[('asm_score', 'std')]])
    eps_data.append(['ASR (%)'] + [f'{v*100:.2f}' for v in by_epsilon[('attack_success', 'mean')]])
    
    table2 = axes[1].table(cellText=eps_data, loc='center', cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 2)
    axes[1].set_title('(B) Statistics by Epsilon', fontweight='bold', pad=20)
    
    # Table 3: By Attack
    axes[2].axis('tight')
    axes[2].axis('off')
    att_data = []
    att_data.append(['Attack'] + [a.upper() for a in by_attack.index])
    att_data.append(['Mean ASM'] + [f'{v:.4f}' for v in by_attack[('asm_score', 'mean')]])
    att_data.append(['Std ASM'] + [f'{v:.4f}' for v in by_attack[('asm_score', 'std')]])
    att_data.append(['ASR (%)'] + [f'{v*100:.2f}' for v in by_attack[('attack_success', 'mean')]])
    
    table3 = axes[2].table(cellText=att_data, loc='center', cellLoc='center')
    table3.auto_set_font_size(False)
    table3.set_fontsize(10)
    table3.scale(1, 2)
    axes[2].set_title('(C) Statistics by Attack Type', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Summary Table: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate publication-ready figures')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing GAP-4 results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for figures (default: results_dir/figures)')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    df, stats_data = load_results(args.results_dir)
    print(f"Loaded {len(df)} samples")
    
    # Setup output directory
    if args.output_dir is None:
        output_dir = Path(args.results_dir) / 'figures'
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to: {output_dir}")
    
    # Generate figures
    print("\n" + "="*60)
    print("Generating Publication Figures")
    print("="*60 + "\n")
    
    figure_1_asm_overview(df, output_dir / 'figure1_asm_overview.png')
    figure_2_attack_comparison(df, output_dir / 'figure2_attack_comparison.png')
    figure_3_epsilon_analysis(df, output_dir / 'figure3_epsilon_analysis.png')
    generate_summary_table(df, stats_data, output_dir / 'table_summary.png')
    
    print("\n" + "="*60)
    print("Publication Figures Complete!")
    print("="*60)
    print(f"\nFigures saved to: {output_dir}")
    print("\nGenerated:")
    print("  - figure1_asm_overview.png")
    print("  - figure2_attack_comparison.png")
    print("  - figure3_epsilon_analysis.png")
    print("  - table_summary.png")


if __name__ == '__main__':
    main()

