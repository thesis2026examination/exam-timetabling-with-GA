import argparse
import logging
import os

from src.genetic_algorithm import GeneticAlgorithm
from src.parser import load_csv_dataset

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="CSV Exam Timetabling Genetic Algorithm")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="archive",
        help="Directory containing classrooms.csv, courses.csv, instructors.csv, schedule.csv, students.csv, timeslots.csv",
    )
    parser.add_argument("--pop-size", type=int, default=50, help="Population size")
    parser.add_argument("--gens", type=int, default=100, help="Number of generations")
    parser.add_argument("--mut-rate", type=float, default=0.15, help="Mutation probability")
    parser.add_argument("--tournament-size", type=int, default=5, help="Tournament selection size")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.data_dir):
        logging.error("Data directory not found: %s", args.data_dir)
        return
        
    logging.info("Loading CSV dataset")
    dataset = load_csv_dataset(args.data_dir)
    logging.info(
        "Loaded %s courses, %s classrooms, %s instructors, %s timeslots, %s students.",
        len(dataset.courses),
        len(dataset.classrooms),
        len(dataset.instructors),
        len(dataset.timeslots),
        len(dataset.students),
    )
    logging.info("Conflict matrix shape: %s", dataset.conflict_matrix.shape)
    logging.info(
        "Enrollment range: min=%s max=%s",
        min(dataset.course_enrollment_counts.values()),
        max(dataset.course_enrollment_counts.values()),
    )

    ga = GeneticAlgorithm(
        dataset,
        population_size=args.pop_size,
        mutation_rate=args.mut_rate,
        tournament_size=args.tournament_size,
    )
    fitness, penalty, scores, chromosome = ga.run_with_scores(generations=args.gens)
    logging.info("Best chromosome: %s", chromosome)
    logging.info("Best fitness: %.10f", fitness)
    logging.info("Best total penalty: %s", penalty)
    logging.info("Best score breakdown: %s", scores)
            
if __name__ == "__main__":
    main()
