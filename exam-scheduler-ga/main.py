import argparse
import logging
import os
from src.parser import parse_dataset
from src.genetic_algorithm import GeneticAlgorithm

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Exam Timetabling using Genetic Algorithm")
    parser.add_argument("--dataset", type=str, default="c:/exam-timetabling-with-GA/pu-exam-fal10.xml",
                        help="Path to the XML dataset file")
    parser.add_argument("--pop-size", type=int, default=10, help="Population size")
    parser.add_argument("--gens", type=int, default=50, help="Number of generations")
    parser.add_argument("--mut-rate", type=float, default=0.1, help="Mutation rate")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        logging.error(f"Dataset file not found: {args.dataset}")
        return
        
    logging.info("Step 1: Parsing Dataset")
    dataset = parse_dataset(args.dataset)
    logging.info(f"Loaded {len(dataset.exams)} exams, {len(dataset.rooms)} rooms, "
                 f"{len(dataset.periods)} periods, {len(dataset.student_exams)} students.")
    logging.info(f"Identified {len(dataset.large_exams_indices)} large exams and "
                 f"{len(dataset.time_fixed_exams)} time-fixed exams.")
                 
    logging.info("Step 2: Initializing Genetic Algorithm")
    ga = GeneticAlgorithm(dataset, population_size=args.pop_size, mutation_rate=args.mut_rate)
    
    logging.info("Step 3: Running Genetic Algorithm Optimization")
    best_chromosome = ga.run(generations=args.gens)
    
    logging.info("Optimization complete!")
    best_fitness, scores = ga.fitness_calc.calculate_fitness(best_chromosome)
    logging.info(f"Best Fitness (Penalty): {best_fitness}")
    for k, v in scores.items():
        if v > 0:
            logging.info(f"  - {k}: {v}")
            
if __name__ == "__main__":
    main()
