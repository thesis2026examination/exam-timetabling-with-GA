import argparse
import logging
import os
import sys

from src.genetic_algorithm import GeneticAlgorithm
from src.dataset_parser import parse_xml_dataset

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    parser = argparse.ArgumentParser(description="ITC XML Exam Timetabling Pure Genetic Algorithm")
    parser.add_argument(
        "--xml-file",
        type=str,
        default="aghfal17_postcompetition2.xml",
        help="Path to the ITC XML dataset file"
    )
    parser.add_argument("--pop-size", type=int, default=30, help="Population size")
    parser.add_argument("--gens", type=int, default=50, help="Number of generations")
    parser.add_argument("--mut-rate", type=float, default=0.15, help="Mutation probability")
    parser.add_argument("--tournament-size", type=int, default=5, help="Tournament selection size")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.xml_file):
        logging.error("XML dataset file not found: %s", args.xml_file)
        return
        
    logging.info("Loading XML dataset...")
    dataset = parse_xml_dataset(args.xml_file)
    logging.info(
        "Loaded XML Dataset '%s' | %s classes, %s rooms, %s students, %s distributions.",
        dataset.name,
        len(dataset.classes),
        len(dataset.rooms),
        len(dataset.students),
        len(dataset.distributions)
    )
    logging.info(f"Total time slots in timeline: {dataset.total_slots} slots.")

    ga = GeneticAlgorithm(
        dataset,
        population_size=args.pop_size,
        mutation_rate=args.mut_rate,
        tournament_size=args.tournament_size,
    )
    fitness, penalty, scores, chromosome = ga.run_with_scores(generations=args.gens)
    
    logging.info("=========================================")
    logging.info("             RUN COMPLETED               ")
    logging.info("=========================================")
    logging.info("Best fitness: %.10f", fitness)
    logging.info("Best total penalty: %d", penalty)
    logging.info("Penalty Breakdown:")
    logging.info("  Student conflicts:      %d (Weight: %.1f)", scores["student"], dataset.weights.get("student", 5.0))
    logging.info("  Room violations:       %d (Weight: %.1f)", scores["room"], dataset.weights.get("room", 1.0))
    logging.info("  Forbidden time slots:  %d (Weight: %.1f)", scores["time"], dataset.weights.get("time", 4.0))
    logging.info("  Distribution rule violations: %d (Weight: %.1f)", scores["distribution"], dataset.weights.get("distribution", 15.0))
            
if __name__ == "__main__":
    main()
