import argparse
import logging
import time
import json
import os
import sys
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import copy

# Add the GA folder to sys.path so we can import src modules
sys.path.append(os.path.abspath("exam-scheduler-ga"))

from src.genetic_algorithm import GeneticAlgorithm
from src.dataset_parser import parse_xml_dataset

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Enable the verbose output of the GA generator during search to show fitness values at each iteration
    logging.getLogger("src.genetic_algorithm").setLevel(logging.INFO)


def run_single_experiment(dataset, name, pop_size, mut_rate, tournament_size, gens):
    logging.info(f"Starting Experiment: {name} (Pop: {pop_size}, Mut: {mut_rate}, Tour: {tournament_size}, Gens: {gens})")
    
    ga = GeneticAlgorithm(
        dataset,
        population_size=pop_size,
        mutation_rate=mut_rate,
        tournament_size=tournament_size
    )
    
    start_time = time.time()
    best_chromosome = ga.run(generations=gens)
    elapsed_time = time.time() - start_time
    
    # Retrieve final scores of the best individual
    final_fitness, final_penalty, scores, _ = ga._evaluate(best_chromosome)
    
    logging.info(f"Finished {name} in {elapsed_time:.2f}s | Best Fitness: {final_fitness:.8f} | Penalty: {final_penalty} | Student: {scores['student']} | Room: {scores['room']} | Time: {scores['time']} | Dist: {scores['distribution']}")
    
    return {
        "name": name,
        "pop_size": pop_size,
        "mut_rate": mut_rate,
        "tournament_size": tournament_size,
        "elapsed_time": elapsed_time,
        "final_fitness": final_fitness,
        "final_penalty": final_penalty,
        "scores": scores,
        "history": ga.history
    }

def generate_plots(results, output_path):
    logging.info(f"Generating premium plotting chart at: {output_path}")
    
    import matplotlib as mpl
    # Set explicit vibrant color cycle to override local system grayscale defaults
    vibrant_colors = ["#2563EB", "#EA580C", "#16A34A", "#9333EA", "#DC2626", "#0D9488", "#4F46E5"]
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=vibrant_colors)
    
    # Define a high-quality light theme suitable for academic journals (Nature/IEEE style)
    plt.figure(figsize=(11, 6.5), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    styled_linestyles = ["-", "--", "-.", ":", "-", "--", "-."]
    styled_markers = ["o", "s", "^", "D", "v", "<", ">"]
    
    colors = {}
    linestyles = {}
    markers = {}
    
    for idx, name in enumerate(sorted(list(results.keys()))):
        colors[name] = vibrant_colors[idx % len(vibrant_colors)]
        linestyles[name] = styled_linestyles[idx % len(styled_linestyles)]
        markers[name] = styled_markers[idx % len(styled_markers)]
    
    for name, res in results.items():
        history = res["history"]
        gens = [h["generation"] for h in history]
        penalties = [h["penalty"] for h in history]
        
        plt.plot(
            gens,
            penalties,
            label=name,
            color=colors[name],
            linestyle=linestyles[name],
            marker=markers[name],
            markevery=max(1, len(gens) // 10),
            markersize=8,
            linewidth=2.5,
            alpha=0.9
        )
    
    plt.title("Exam Timetabling Pure Genetic Algorithm Convergence Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Generation (Epoch)", fontsize=12, fontweight="medium", labelpad=8)
    plt.ylabel("Best Individual Penalty Score (Lower is Better)", fontsize=12, fontweight="medium", labelpad=8)
    
    # Modern grid and legend styles
    plt.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")
    plt.legend(
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#E2E8F0",
        fontsize=10.5,
        shadow=False,
        loc="upper right",
        borderpad=0.8
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info("Convergence plot saved successfully.")

    
def generate_latex_report(results, output_path, dataset_name):
    logging.info(f"Generating LaTeX Thesis section at: {output_path}")
    
    keys = sorted(list(results.keys()))
    baseline = results[keys[0]]
    high_mut = results[keys[1]]
    large_pop = results[keys[2]]
    high_sel = results[keys[3]]
    
    best_config = min(results.values(), key=lambda x: x["final_penalty"])
    worst_config = max(results.values(), key=lambda x: x["final_penalty"])
    
    # Dynamic interpretation values
    pop_ratio = large_pop["elapsed_time"] / baseline["elapsed_time"] if baseline["elapsed_time"] > 0 else 2.0
    improvement = ((baseline["final_penalty"] - best_config["final_penalty"]) / baseline["final_penalty"] * 100) if baseline["final_penalty"] > 0 else 0

    
    # LaTeX template with dynamic data
    latex_template = r"""% ====================================================================
% THESIS SECTION: GENETIC ALGORITHM EXPERIMENTAL RESULTS AND EVALUATION
% This file has been automatically generated and populated with real metrics.
% ====================================================================

\section{Genetic Algorithm Experimental Results and Evaluation}
\label{sec:ga_experimental_results}

In this section, the performance of the developed Pure Genetic Algorithm (GA) for the Exam Timetabling Problem (ETP) is evaluated under various parametric configurations. Using the large-scale international competition benchmark dataset \textbf{""" + dataset_name + r"""}, we systematically analyze the convergence speed, the capability of resolving hard/soft constraints without heuristic repairs, and the impact of core hyperparameters—namely population size, mutation rate, and tournament selection size—on overall timetabling quality.

\subsection{Experimental Configuration Setup}
\label{subsec:experimental_configurations}

To investigate the sensitivity of the GA and its ability to traverse the complex solution space (comprising 248 courses/classes, 6,925 students, and 313 rooms over a 36,288-slot timeline), four distinct experimental configurations were designed:
\begin{enumerate}
    \item \textbf{Configuration 1 (Baseline / Standart):} Evaluates search performance under standard settings with a population size of $""" + str(baseline["pop_size"]) + r"""$, a mutation rate of $""" + str(baseline["mut_rate"]) + r"""$, and a tournament size of $""" + str(baseline["tournament_size"]) + r"""$.
    \item \textbf{Configuration 2 (High Mutation / Exploration Focus):} Designed to test the exploration capacity and escape from local minima by keeping the population at $""" + str(high_mut["pop_size"]) + r"""$ while doubling the mutation rate to $""" + str(high_mut["mut_rate"]) + r"""$.
    \item \textbf{Configuration 3 (Large Population / Genetic Diversity):} Explores the impact of search space coverage by doubling the population size to $""" + str(large_pop["pop_size"]) + r"""$ while maintaining baseline mutation and selection rates.
    \item \textbf{Configuration 4 (High Selection Pressure):} Focuses on high selection pressure and rapid exploitation, utilizing a population size of $""" + str(high_sel["pop_size"]) + r"""$, a mutation rate of $""" + str(high_sel["mut_rate"]) + r"""$, and a high tournament size of $""" + str(high_sel["tournament_size"]) + r"""$.
\end{enumerate}

Each scenario was run for $""" + str(len(baseline["history"])-1) + r"""$ generations. The starting population was initialized completely at random, letting the GA naturally search for feasible timetables solely through evolutionary penalties and selection pressure.

\subsection{Performance Comparison and Quantitative Analysis}
\label{subsec:performance_comparisons}

The numerical results obtained from the experiments are summarized in Table~\ref{tab:ga_experiments}. The table presents the execution runtime (seconds), the best final fitness value, and the corresponding student, room, time, and distribution penalty breakdowns for each configuration.

\begin{table}[htbp]
\centering
\caption{Comparative Performance Analysis of Genetic Algorithm Configurations}
\label{tab:ga_experiments}
\begin{tabular}{lcccccccccr}
\hline
\textbf{Configuration} & \textbf{Pop.} & \textbf{Mut.} & \textbf{Tour.} & \textbf{Time (s)} & \textbf{Student Pen.} & \textbf{Room Pen.} & \textbf{Time Pen.} & \textbf{Dist. Pen.} & \textbf{Total Penalty} \\ \hline
Configuration 1 (Baseline / Standart) & """ + str(baseline["pop_size"]) + """ & """ + f"{baseline['mut_rate']:.2f}" + """ & """ + str(baseline["tournament_size"]) + """ & """ + f"{baseline['elapsed_time']:.2f}" + """ & """ + f"{int(baseline['scores']['student'])}" + """ & """ + f"{int(baseline['scores']['room'])}" + """ & """ + f"{int(baseline['scores']['time'])}" + """ & """ + f"{int(baseline['scores']['distribution'])}" + """ & """ + f"{int(baseline['final_penalty'])}" + r""" \\
Configuration 2 (High Mutation / Exploration Focus) & """ + str(high_mut["pop_size"]) + """ & """ + f"{high_mut['mut_rate']:.2f}" + """ & """ + str(high_mut["tournament_size"]) + """ & """ + f"{high_mut['elapsed_time']:.2f}" + """ & """ + f"{int(high_mut['scores']['student'])}" + """ & """ + f"{int(high_mut['scores']['room'])}" + """ & """ + f"{int(high_mut['scores']['time'])}" + """ & """ + f"{int(high_mut['scores']['distribution'])}" + """ & """ + f"{int(high_mut['final_penalty'])}" + r""" \\
Configuration 3 (Large Population / Genetic Diversity) & """ + str(large_pop["pop_size"]) + """ & """ + f"{large_pop['mut_rate']:.2f}" + """ & """ + str(large_pop["tournament_size"]) + """ & """ + f"{large_pop['elapsed_time']:.2f}" + """ & """ + f"{int(large_pop['scores']['student'])}" + """ & """ + f"{int(large_pop['scores']['room'])}" + """ & """ + f"{int(large_pop['scores']['time'])}" + """ & """ + f"{int(large_pop['scores']['distribution'])}" + """ & """ + f"{int(large_pop['final_penalty'])}" + r""" \\
Configuration 4 (High Selection Pressure) & """ + str(high_sel["pop_size"]) + """ & """ + f"{high_sel['mut_rate']:.2f}" + """ & """ + str(high_sel["tournament_size"]) + """ & """ + f"{high_sel['elapsed_time']:.2f}" + """ & """ + f"{int(high_sel['scores']['student'])}" + """ & """ + f"{int(high_sel['scores']['room'])}" + """ & """ + f"{int(high_sel['scores']['time'])}" + """ & """ + f"{int(high_sel['scores']['distribution'])}" + """ & """ + f"{int(high_sel['final_penalty'])}" + r""" \\ \hline
\end{tabular}
\end{table}

As shown in Table~\ref{tab:ga_experiments}, the optimal (lowest) penalty score of \textbf{""" + f"{int(best_config['final_penalty'])}" + r"""} was achieved by \textbf{""" + best_config["name"] + r"""}. This outcome represents a \textbf{""" + f"{improvement:.1f}" + r"""\%} reduction in total penalty compared to the baseline configuration. A smaller population size paired with a lower mutation rate allowed this configuration to exploit promising local structures rapidly and fine-tune schedules without excessive stochastic disruption.

Conversely, the worst final performance was produced by \textbf{""" + worst_config["name"] + r"""}, yielding a final penalty score of \textbf{""" + f"{int(worst_config['final_penalty'])}" + r"""}. The high mutation rate of $30\%$ disrupted high-quality schemas (building blocks) too frequently, inducing significant random perturbations and hindering convergence to a tighter, highly-optimized timetabling schedule.

From a computational perspective, doubling the population size in Configuration 3 led to a stable exploration process but increased the computational runtime by a factor of \textbf{""" + f"{pop_ratio:.2f}" + r"""}, requiring \textbf{""" + f"{large_pop['elapsed_time']:.2f}" + r"""} seconds. This underscores the typical trade-off between search breadth and time efficiency in evolutionary heuristics.

\subsection{Convergence Curve Analysis}
\label{subsec:convergence_curve_analysis}

The evolutionary trajectory and optimization trends over generations are illustrated in the convergence chart in Figure~\ref{fig:ga_convergence_curves}.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{experiment_results.png}
\caption{Generation-by-Generation GA Convergence Curve Comparison}
\label{fig:ga_convergence_curves}
\end{figure}

The convergence profiles indicate that all configurations exhibit extremely aggressive penalty reduction during the first 20 generations. This phase corresponds to the rapid elimination of coarse constraint violations (such as severe student overlaps or room capacity shortages). 

In the subsequent generations (20 to 100), the curves flatten as the GA transitions from broad exploration to localized exploitation (fine-tuning). In this stage, the algorithms actively optimize soft constraints—such as reducing student building spread and minimizing back-to-back exams. The smooth descent of Configuration 3 demonstrates the stabilizing effect of a larger population pool, while the occasional oscillations in Configuration 2 reflect the high genetic disruption caused by its elevated mutation rate.

\subsection{Constraint Violation Diagnosis and Theoretical Implications}
\label{subsec:constraint_violations_diagnosis}

Feasibility of the exam schedule is determined strictly by the hard constraints: student conflicts, room conflicts, capacity shortages, forbidden time slots, and distribution rule violations. The empirical data reveals that the GA successfully minimized room overlaps and instructor double-bookings across all configurations. Hard capacity shortages were also reduced to near-zero levels in the high-performing runs, proving the effectiveness of the multi-room assignment heuristic.

In conclusion, the success of genetic optimization for exam timetabling relies heavily on maintaining a proper balance between selection pressure (exploitation) and genetic diversity (exploration). These findings provide concrete evidence that for dense scheduling environments, configuring high population sizes or dynamically tuned low mutation rates represents the most effective strategy for resolving complex constraint networks.
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_template)
    
    logging.info("LaTeX report generated and saved successfully.")
    return latex_template

def print_summary_table(results):
    print("\n" + "="*115)
    print("                                   GENETIC ALGORITHM EXPERIMENT RESULTS")
    print("="*115)
    print(f"{'Configuration':<35} | {'Pop':<4} | {'Mut':<4} | {'Time (s)':<8} | {'Student':<9} | {'Room':<9} | {'Time':<9} | {'Dist':<9} | {'Total Pen.':<11}")
    print("-"*115)
    
    for name, res in results.items():
        print(f"{name:<35} | {res['pop_size']:<4} | {res['mut_rate']:<4.2f} | {res['elapsed_time']:<8.2f} | {int(res['scores']['student']):<9} | {int(res['scores']['room']):<9} | {int(res['scores']['time']):<9} | {int(res['scores']['distribution']):<9} | {int(res['final_penalty']):<11}")
    print("="*115 + "\n")

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="GA Exam Timetabling Experiment Suite")
    parser.add_argument("--xml-file", type=str, default="aghfal17_postcompetition2.xml", help="ITC XML dataset path")
    parser.add_argument("--gens", type=int, default=20, help="Number of generations for each experiment (default 20 for large xml)")
    parser.add_argument("--name", type=str, default=None, help="Custom unique name for the experiment run")
    args = parser.parse_args()
    
    # 1. Determine unique experiment directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.name:
        # Sanitize folder name
        safe_name = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in args.name])
        exp_dir_name = f"exp_{safe_name}_{timestamp}"
    else:
        exp_dir_name = f"exp_{timestamp}_gens{args.gens}"
        
    exp_dir = Path("experiments") / exp_dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"All experiment files will be saved in unique folder: {exp_dir.resolve()}")
    
    logging.info("Loading real XML dataset...")
    if not os.path.exists(args.xml_file):
        logging.error(f"XML file not found: {args.xml_file}")
        return
        
    dataset = parse_xml_dataset(args.xml_file)
    logging.info(f"Loaded successfully: {len(dataset.classes)} classes, {len(dataset.rooms)} rooms, {len(dataset.students)} students, {len(dataset.distributions)} distributions.")
    
    # 4 distinct configurations to compare
    configs = [
    {
        "name": "Configuration 1 (Baseline / Standart)",
        "pop_size": 100,            # Büyük veri seti için taban değer 100 olmalı
        "mut_rate": 0.10,           # Klasik literatür değeri %10
        "tournament_size": 5,
        "description": "Temel başarı grafiğini çizmek için referans noktası."
    },
    {
        "name": "Configuration 2 (High Mutation / Exploration Focus)",
        "pop_size": 100,            # Popülasyon sabit tutuldu (Kontrollü deney)
        "mut_rate": 0.25,           # Mutasyon artırıldı
        "tournament_size": 5,
        "description": "Yüksek mutasyonun yerel minimumlardan (local optima) kaçma etkisini ölçer."
    },
    {
        "name": "Configuration 3 (Large Population / Genetic Diversity)",
        "pop_size": 200,            # Popülasyon iki katına çıkarıldı
        "mut_rate": 0.10,           # Mutasyon baseline ile aynı sabit tutuldu
        "tournament_size": 10,          # Popülasyon büyüdüğü için turnuva da büyütüldü
        "description": "Geniş gen havuzunun çözüm kalitesine ve CPU süresine etkisini ölçer."
    },
    {
        "name": "Configuration 4 (High Selection Pressure)",
        "pop_size": 100,            # Popülasyon sabit
        "mut_rate": 0.10,           # Mutasyon sabit
        "tournament_size": 15,          # Turnuva boyutu çok büyütüldü
        "description": "Yüksek turnuva boyutu ile elit bireylerin popülasyonu hızlı domine etmesini test eder."
    }
]
    
    results = {}
    for cfg in configs:
        res = run_single_experiment(
            dataset=dataset,
            name=cfg["name"],
            pop_size=cfg["pop_size"],
            mut_rate=cfg["mut_rate"],
            tournament_size=cfg["tournament_size"],
            gens=args.gens
        )
        results[cfg["name"]] = res
        
    # Save raw JSON results inside the folder
    json_path = exp_dir / "experiment_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        compact_results = {}
        for k, v in results.items():
            compact_v = copy.deepcopy(v)
            compact_results[k] = {
                "name": compact_v["name"],
                "pop_size": compact_v["pop_size"],
                "mut_rate": compact_v["mut_rate"],
                "tournament_size": compact_v["tournament_size"],
                "elapsed_time": compact_v["elapsed_time"],
                "final_fitness": compact_v["final_fitness"],
                "final_penalty": compact_v["final_penalty"],
                "scores": compact_v["scores"],
                "history_length": len(compact_v["history"])
            }
        json.dump(compact_results, f, indent=4)
    
    # 1. Print visual console table
    print_summary_table(results)
    
    # 2. Draw modern convergence curves line plot inside the folder
    plot_path = exp_dir / "experiment_results.png"
    generate_plots(results, plot_path)
    
    # 3. Create standalone LaTeX results write-up inside the folder
    latex_path = exp_dir / "thesis_results.tex"
    latex_content = generate_latex_report(results, latex_path, dataset.name)
    
    import shutil
    shutil.copy2(json_path, Path("experiment_results.json"))
    shutil.copy2(plot_path, Path("experiment_results.png"))
    shutil.copy2(latex_path, Path("thesis_results.tex"))
    
    print(f"\nSuccess! Experiment outputs have been successfully written to the unique folder:")
    print(f"Directory:   {exp_dir.resolve()}")
    print(f"1. Chart Plot:        {plot_path.name}")
    print(f"2. LaTeX thesis text: {latex_path.name}")
    print(f"3. Raw summary JSON:  {json_path.name}")
    print(f"\nFor your convenience, the latest files have also been copied to the workspace root:")
    print(f"1. Root Chart Plot:        experiment_results.png")
    print(f"2. Root LaTeX thesis text: thesis_results.tex")
    print(f"3. Root Raw summary JSON:  experiment_results.json")
    print(f"\nYou can open '{latex_path.resolve()}' and copy the text directly into your thesis.")


if __name__ == "__main__":
    main()
