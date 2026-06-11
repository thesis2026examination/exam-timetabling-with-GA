import os
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    checkpoint_path = "checkpoint.pkl"
    if not os.path.exists(checkpoint_path):
        # Check in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(script_dir, "checkpoint.pkl")
        if not os.path.exists(checkpoint_path):
            logging.error("checkpoint.pkl not found! Please run an experiment first or provide the correct path.")
            return

    logging.info(f"Loading checkpoint from {checkpoint_path}...")
    with open(checkpoint_path, "rb") as f:
        population, history = pickle.load(f)

    if not history:
        logging.error("No history found in the checkpoint!")
        return

    logging.info(f"Loaded history with {len(history)} generations.")

    # Extract data
    gens = [h["generation"] for h in history]
    student_penalties = [h.get("student", 0) for h in history]
    room_penalties = [h.get("room", 0) for h in history]
    time_penalties = [h.get("time", 0) for h in history]
    dist_penalties = [h.get("dist", 0) for h in history]
    total_penalties = [h.get("penalty", 0) for h in history]

    # Setup styling for academic publication
    vibrant_colors = ["#2563EB", "#EA580C", "#16A34A", "#9333EA", "#DC2626"] # Blue, Orange, Green, Purple, Red
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=vibrant_colors)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # --- Plot 1: Combined Constraints Convergence (Multi-line) ---
    plt.figure(figsize=(11, 6.5), dpi=300)
    
    # Plot each constraint
    plt.plot(gens, student_penalties, label="Student Conflict Penalty", color="#2563EB", linewidth=2, alpha=0.85)
    plt.plot(gens, room_penalties, label="Room Capacity/Overlap Penalty", color="#EA580C", linewidth=2, alpha=0.85)
    plt.plot(gens, time_penalties, label="Time Constraint Penalty", color="#16A34A", linewidth=2, alpha=0.85)
    plt.plot(gens, dist_penalties, label="Distribution Rule Penalty", color="#9333EA", linewidth=2, alpha=0.85)
    plt.plot(gens, total_penalties, label="Total Penalty (Sum)", color="#DC2626", linewidth=2.5, linestyle="--", alpha=0.9)

    plt.title("Exam Timetabling Constraints Convergence Trajectory", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Generation (Epoch)", fontsize=12, fontweight="medium", labelpad=8)
    plt.ylabel("Penalty Score (Lower is Better)", fontsize=12, fontweight="medium", labelpad=8)
    
    plt.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")
    plt.legend(
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#E2E8F0",
        fontsize=10.5,
        loc="upper right",
        borderpad=0.8
    )
    plt.tight_layout()
    combined_plot_path = "constraints_convergence_combined.png"
    plt.savefig(combined_plot_path, dpi=300)
    plt.close()
    logging.info(f"Saved combined plot to {combined_plot_path}")

    # --- Plot 2: Combined Constraints Convergence (Log Scale) ---
    plt.figure(figsize=(11, 6.5), dpi=300)
    
    # Use log scale for Y-axis to handle large scale differences
    plt.semilogy(gens, student_penalties, label="Student Conflict Penalty", color="#2563EB", linewidth=2, alpha=0.85)
    plt.semilogy(gens, room_penalties, label="Room Capacity/Overlap Penalty", color="#EA580C", linewidth=2, alpha=0.85)
    plt.semilogy(gens, time_penalties, label="Time Constraint Penalty", color="#16A34A", linewidth=2, alpha=0.85)
    plt.semilogy(gens, dist_penalties, label="Distribution Rule Penalty", color="#9333EA", linewidth=2, alpha=0.85)
    plt.semilogy(gens, total_penalties, label="Total Penalty (Sum)", color="#DC2626", linewidth=2.5, linestyle="--", alpha=0.9)

    plt.title("Exam Timetabling Constraints Convergence (Logarithmic Scale)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Generation (Epoch)", fontsize=12, fontweight="medium", labelpad=8)
    plt.ylabel("Penalty Score (Log Scale, Lower is Better)", fontsize=12, fontweight="medium", labelpad=8)
    
    plt.grid(True, which="both", linestyle="--", alpha=0.5, color="#CBD5E1")
    plt.legend(
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#E2E8F0",
        fontsize=10.5,
        loc="upper right",
        borderpad=0.8
    )
    plt.tight_layout()
    log_plot_path = "constraints_convergence_log.png"
    plt.savefig(log_plot_path, dpi=300)
    plt.close()
    logging.info(f"Saved log-scale plot to {log_plot_path}")

    # --- Plot 3: 2x2 Subplots (Individual Constraints) ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.suptitle("Individual Constraint Penalty Convergence", fontsize=16, fontweight="bold", y=0.98)

    # 1. Student Penalty
    axs[0, 0].plot(gens, student_penalties, color="#2563EB", linewidth=2)
    axs[0, 0].set_title("Student Conflict Penalty", fontsize=12, fontweight="semibold")
    axs[0, 0].set_xlabel("Generation", fontsize=10)
    axs[0, 0].set_ylabel("Penalty", fontsize=10)
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)

    # 2. Room Penalty
    axs[0, 1].plot(gens, room_penalties, color="#EA580C", linewidth=2)
    axs[0, 1].set_title("Room Capacity/Overlap Penalty", fontsize=12, fontweight="semibold")
    axs[0, 1].set_xlabel("Generation", fontsize=10)
    axs[0, 1].set_ylabel("Penalty", fontsize=10)
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)

    # 3. Time Penalty
    axs[1, 0].plot(gens, time_penalties, color="#16A34A", linewidth=2)
    axs[1, 0].set_title("Time Constraint Penalty", fontsize=12, fontweight="semibold")
    axs[1, 0].set_xlabel("Generation", fontsize=10)
    axs[1, 0].set_ylabel("Penalty", fontsize=10)
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)

    # 4. Distribution Penalty
    axs[1, 1].plot(gens, dist_penalties, color="#9333EA", linewidth=2)
    axs[1, 1].set_title("Distribution Rule Penalty", fontsize=12, fontweight="semibold")
    axs[1, 1].set_xlabel("Generation", fontsize=10)
    axs[1, 1].set_ylabel("Penalty", fontsize=10)
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    subplots_path = "constraints_convergence_subplots.png"
    plt.savefig(subplots_path, dpi=300)
    plt.close()
    logging.info(f"Saved subplots to {subplots_path}")

    print("\n" + "="*80)
    print("SUCCESSFULLY GENERATED CONSTRAINT CONVERGENCE CHARTS")
    print("="*80)
    print(f"1. Combined linear chart:   {combined_plot_path}")
    print(f"2. Combined log-scale chart: {log_plot_path}")
    print(f"3. Individual 2x2 subplots:  {subplots_path}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
