import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import json
import os
from datetime import datetime
from AIMM_simulator import *

class EXP3CellOnOff(Scenario):
    """
    EXP3 algorithm-based cell on/off scenario.
    Selects n cells out of k cells to turn off based on network efficiency (throughput/power).
    """
    
    def __init__(self, sim, k_cells=None, n_cells_off=5, interval=1.0, 
                 eta=0.1, warm_up=True, warm_up_episodes=100, 
                 output_dir=None, delay=0.0):
        """
        Initializes an instance of the EXP3CellOnOff class.
        
        Parameters:
        -----------
        sim : SimPy.Environment
            The simulation environment object.
        k_cells : list of int, optional
            List of cell indices to consider. Default is all cells.
        n_cells_off : int
            Number of cells to turn off.
        interval : float
            Time interval between each decision.
        eta : float
            Learning rate for EXP3 algorithm.
        warm_up : bool
            Whether to use epsilon-greedy warm-up.
        warm_up_episodes : int
            Number of warm-up episodes.
        output_dir : str
            Directory to save results.
        delay : float
            Initial delay before starting.
        """
        self.sim = sim
        self.interval = interval
        self.delay_time = delay
        self.n_cells_off = n_cells_off
        self.eta = eta
        self.warm_up = warm_up
        self.warm_up_episodes = warm_up_episodes
        self.episode = 0
        
        # Setup output directory
        if output_dir is None:
            self.output_dir = f"data/output/exp3_cell_onoff_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Determine cells to consider
        if k_cells is None:
            self.k_cells = list(range(len(self.sim.cells)))
        else:
            self.k_cells = k_cells
        
        # Generate all possible combinations of cells to turn off
        self.arms = list(combinations(self.k_cells, self.n_cells_off))
        self.n_arms = len(self.arms)
        
        # Initialize weights and probabilities
        self.weights = np.ones(self.n_arms)
        self.probabilities = np.ones(self.n_arms) / self.n_arms
        
        # Initialize tracking variables
        self.arm_selections = []
        self.rewards = []
        self.weights_history = []
        self.probabilities_history = []
        self.efficiency_history = []
        
        # Current state
        self.current_arm = None
        self.cells_currently_off = set()
        
        # Energy model references
        self.cell_energy_models = {}
        
        print(f"EXP3CellOnOff initialized:")
        print(f"  - Total cells: {len(self.sim.cells)}")
        print(f"  - Cells to consider: {len(self.k_cells)}")
        print(f"  - Cells to turn off: {self.n_cells_off}")
        print(f"  - Number of arms: {self.n_arms}")
        print(f"  - Learning rate (eta): {self.eta}")
        print(f"  - Warm-up: {self.warm_up} ({self.warm_up_episodes} episodes)")
        
    def setup_energy_models(self, energy_models_dict):
        """Set reference to energy models dictionary."""
        self.cell_energy_models = energy_models_dict
        
    def calculate_network_efficiency(self):
        """Calculate current network efficiency (throughput/power)."""
        total_throughput = 0.0
        total_power = 0.0
        
        print(f"Debug: Calculating efficiency at t={self.sim.env.now}")

        for cell_idx in self.k_cells:
            cell = self.sim.cells[cell_idx]
            
            # Get cell throughput
            cell_throughput = cell.get_cell_throughput()  # Mbps
            total_throughput += cell_throughput
            
            # Get cell power consumption - 수정된 부분
            try:
                power_dBm = cell.get_power_dBm()
                
                # Cell이 OFF 상태인지 확인
                if power_dBm <= -100 or np.isinf(power_dBm):
                    cell_power = 0.1  # 최소 대기전력 (kW)
                else:
                    if cell_idx in self.cell_energy_models:
                        cell_power_watts = self.cell_energy_models[cell_idx].get_cell_power_watts(self.sim.env.now)
                        # nan 체크
                        if np.isnan(cell_power_watts) or np.isinf(cell_power_watts):
                            cell_power = 2.0  # 기본값 (kW)
                        else:
                            cell_power = cell_power_watts / 1000.0  # kW로 변환
                    else:
                        cell_power = 2.0  # 기본 ON 상태 전력 (kW)
                
                total_power += cell_power
                
            except Exception as e:
                print(f"Error calculating power for cell {cell_idx}: {e}")
                # 기본값 사용
                cell_power = 0.1 if cell.get_power_dBm() <= -100 else 2.0
                total_power += cell_power
        
        # Calculate efficiency (Mbps/kW)
        if total_power > 0:
            efficiency = total_throughput / total_power
        else:
            efficiency = 0.0
            
        print(f"Efficiency: {efficiency:.2f} Mbps/kW, Throughput: {total_throughput:.2f} Mbps, Power: {total_power:.2f} kW")
        
        return efficiency, total_throughput, total_power


    def select_arm(self):
        """Select an arm using EXP3 algorithm or epsilon-greedy during warm-up."""
        if self.warm_up and self.episode < self.warm_up_episodes:
            # Epsilon-greedy warm-up
            if self.sim.rng.random() < 0.1:  # 10% exploration
                return self.sim.rng.integers(0, self.n_arms)
            else:
                # Random selection during warm-up
                return self.sim.rng.integers(0, self.n_arms)
        else:
            # EXP3 selection based on probabilities
            return self.sim.rng.choice(self.n_arms, p=self.probabilities)
    
    def apply_arm_action(self, arm_idx):
        """Apply the selected arm action (turn cells on/off)."""
        cells_to_turn_off = set(self.arms[arm_idx])
        
        # Turn on cells that should be on
        for cell_idx in self.cells_currently_off - cells_to_turn_off:
            self.sim.cells[cell_idx].set_power_dBm(43.0)  # Default power
            print(f"t={self.sim.env.now:.2f}: Cell[{cell_idx}] turned ON")
        
        # Turn off cells that should be off
        for cell_idx in cells_to_turn_off - self.cells_currently_off:
            self.sim.cells[cell_idx].set_power_dBm(-np.inf)
            print(f"t={self.sim.env.now:.2f}: Cell[{cell_idx}] turned OFF")
        
        self.cells_currently_off = cells_to_turn_off
            
    def update_weights(self, arm_idx, reward):
        """Update EXP3 weights based on reward."""
        if len(self.arm_selections) == 0:
            return
        
        # Reward normalization - 중요!
        normalized_reward = reward / 100.0  # 또는 max_expected_efficiency로 나누기
        
        # Estimated reward for the selected arm
        estimated_reward = normalized_reward / self.probabilities[arm_idx]
        
        # Clip estimated reward to prevent overflow
        estimated_reward = np.clip(estimated_reward, -50, 50)  # exp(±50) 정도가 안전한 범위
        
        # Update weight for selected arm
        weight_update = np.exp(self.eta * estimated_reward)
        
        # Overflow 체크
        if np.isinf(weight_update) or np.isnan(weight_update):
            print(f"Warning: Weight update overflow. estimated_reward={estimated_reward}, using fallback")
            weight_update = np.exp(np.clip(self.eta * estimated_reward, -10, 10))
        
        self.weights[arm_idx] *= weight_update
        
        # Update probabilities with numerical stability
        weight_sum = np.sum(self.weights)
        
        if weight_sum == 0 or np.isnan(weight_sum) or np.isinf(weight_sum):
            print("Warning: Weight sum is invalid, resetting weights")
            self.weights = np.ones(self.n_arms)
            weight_sum = self.n_arms
        
        self.probabilities = self.weights / weight_sum
        
        # Final check for probabilities
        if np.any(np.isnan(self.probabilities)) or np.any(np.isinf(self.probabilities)):
            print("Warning: Probabilities contain NaN/Inf, resetting to uniform")
            self.probabilities = np.ones(self.n_arms) / self.n_arms
            self.weights = np.ones(self.n_arms)
        
    def save_results(self):
        """Save learning results to files."""
        # Save weights history
        weights_file = os.path.join(self.output_dir, "exp3_weights_history.json")
        with open(weights_file, 'w') as f:
            json.dump({
                'weights_history': [w.tolist() for w in self.weights_history],
                'probabilities_history': [p.tolist() for p in self.probabilities_history],
                'arm_selections': self.arm_selections,
                'rewards': self.rewards,
                'efficiency_history': self.efficiency_history,
                'arms': [list(arm) for arm in self.arms]
            }, f, indent=2)
        
        # Plot results
        self.plot_results()
        
    def plot_results(self):
        """Generate plots for learning results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Arm selection probabilities over time
        ax = axes[0, 0]
        if self.probabilities_history:
            probs_array = np.array(self.probabilities_history)
            # Show top 5 arms by final probability
            final_probs = probs_array[-1]
            top_arms = np.argsort(final_probs)[-5:][::-1]
            
            for arm_idx in top_arms:
                ax.plot(probs_array[:, arm_idx], 
                       label=f'Arm {arm_idx}: {self.arms[arm_idx]}')
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Selection Probability')
            ax.set_title('Top 5 Arms Selection Probabilities')
            ax.legend(fontsize=8)
            ax.grid(True)
        
        # Plot 2: Network efficiency over time
        ax = axes[0, 1]
        if self.efficiency_history:
            efficiencies = [e[0] for e in self.efficiency_history]
            ax.plot(efficiencies)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Network Efficiency (Mbps/kW)')
            ax.set_title('Network Efficiency Over Time')
            ax.grid(True)
            
            # Add warm-up boundary
            if self.warm_up:
                ax.axvline(x=self.warm_up_episodes, color='r', linestyle='--', 
                          label='End of warm-up')
                ax.legend()
        
        # Plot 3: Rewards distribution
        ax = axes[1, 0]
        if self.rewards:
            ax.hist(self.rewards, bins=30, alpha=0.7)
            ax.set_xlabel('Reward (Efficiency)')
            ax.set_ylabel('Frequency')
            ax.set_title('Reward Distribution')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Arm selection frequency
        ax = axes[1, 1]
        if self.arm_selections:
            arm_counts = np.bincount(self.arm_selections, minlength=self.n_arms)
            top_arms = np.argsort(arm_counts)[-10:][::-1]
            
            ax.bar(range(len(top_arms)), arm_counts[top_arms])
            ax.set_xlabel('Arm Index')
            ax.set_ylabel('Selection Count')
            ax.set_title('Top 10 Most Selected Arms')
            ax.set_xticks(range(len(top_arms)))
            ax.set_xticklabels(top_arms, rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'exp3_learning_results.png'))
        plt.close()
        
        # Additional plot: Best arm configuration
        self.plot_best_configuration()
        
    def plot_best_configuration(self):
        """Plot the best cell configuration found."""
        if not self.probabilities_history:
            return
            
        # Find best arm by highest probability
        final_probs = self.probabilities_history[-1]
        best_arm_idx = np.argmax(final_probs)
        best_arm = self.arms[best_arm_idx]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Simple visualization showing which cells are on/off
        cell_status = ['ON'] * len(self.sim.cells)
        for cell_idx in best_arm:
            if cell_idx < len(cell_status):
                cell_status[cell_idx] = 'OFF'
        
        # Create bar chart
        colors = ['green' if s == 'ON' else 'red' for s in cell_status]
        bars = ax.bar(range(len(cell_status)), [1]*len(cell_status), color=colors)
        
        ax.set_xlabel('Cell Index')
        ax.set_ylabel('Status')
        ax.set_title(f'Best Configuration (Arm {best_arm_idx}, Prob: {final_probs[best_arm_idx]:.3f})')
        ax.set_ylim(0, 1.5)
        ax.set_yticks([])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='ON'),
                          Patch(facecolor='red', label='OFF')]
        ax.legend(handles=legend_elements)
        
        # Add text showing efficiency
        if self.efficiency_history:
            avg_efficiency = np.mean([e[0] for e in self.efficiency_history[-10:]])
            ax.text(0.02, 0.95, f'Avg Efficiency (last 10): {avg_efficiency:.2f} Mbps/kW',
                   transform=ax.transAxes, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'exp3_best_configuration.png'))
        plt.close()
        
    def loop(self):
        """Main loop of the EXP3 scenario."""
        # Wait for initial delay
        if self.delay_time > 0:
            yield self.sim.env.timeout(self.delay_time)
        
        print(f"EXP3CellOnOff started at t={self.sim.env.now}")
        
        while True:
            # Select arm
            self.current_arm = self.select_arm()
            self.arm_selections.append(self.current_arm)
            
            # Apply arm action
            self.apply_arm_action(self.current_arm)
            
            # Wait for system to stabilize
            yield self.sim.env.timeout(self.interval * 0.5)
            
            # Calculate reward (network efficiency)
            efficiency, throughput, power = self.calculate_network_efficiency()
            reward = efficiency  # Use efficiency as reward
            self.rewards.append(reward)
            self.efficiency_history.append((efficiency, throughput, power))
            
            # Update weights (only after warm-up)
            if not self.warm_up or self.episode >= self.warm_up_episodes:
                self.update_weights(self.current_arm, reward)
            
            # Store history
            self.weights_history.append(self.weights.copy())
            self.probabilities_history.append(self.probabilities.copy())
            
            # Print progress
            if self.episode % 10 == 0:
                print(f"Episode {self.episode}: Arm {self.current_arm}, "
                      f"Reward: {reward:.2f}, Efficiency: {efficiency:.2f} Mbps/kW, "
                      f"Throughput: {throughput:.2f} Mbps, Power: {power:.2f} kW")
            
            # Increment episode
            self.episode += 1
            
            # Save results periodically
            if self.episode % 100 == 0:
                self.save_results()
            
            # Wait for next interval
            yield self.sim.env.timeout(self.interval * 0.5)
            
    def finalize(self):
        """Called at end of simulation."""
        print(f"EXP3CellOnOff finalized after {self.episode} episodes")
        self.save_results()
        
        # Print summary
        if self.probabilities_history:
            final_probs = self.probabilities_history[-1]
            best_arm_idx = np.argmax(final_probs)
            best_arm = self.arms[best_arm_idx]
            
            print("\n=== EXP3 Learning Summary ===")
            print(f"Best arm: {best_arm_idx} with probability {final_probs[best_arm_idx]:.3f}")
            print(f"Cells to turn off: {list(best_arm)}")
            print(f"Average efficiency (last 10): {np.mean([e[0] for e in self.efficiency_history[-10:]]):.2f} Mbps/kW")
            print(f"Results saved to: {self.output_dir}")
