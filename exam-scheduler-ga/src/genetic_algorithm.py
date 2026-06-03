import copy
import logging
import random
from typing import Dict, List, Tuple
from .dataset_parser import XMLDataset
from .fitness import calculate_fitness

logger = logging.getLogger(__name__)

# Chromosome representation: Dict[int, Dict[str, int]]
# { class_id: {"start_slot": int, "room_id": int} }

EvaluatedChromosome = Tuple[float, int, Dict[str, int], Dict[int, Dict[str, int]]]

class GeneticAlgorithm:
    def __init__(
        self,
        dataset: XMLDataset,
        population_size: int = 50,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.90,
        elitism_rate: float = 0.06,
        tournament_size: int = 5
    ):
        self.ds = dataset
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        
        self.population: List[Dict[int, Dict[str, int]]] = []
        self.history: List[Dict] = []
        
    def generate_random_chromosome(self) -> Dict[int, Dict[str, int]]:
        chromosome = {}
        for class_id, cl in self.ds.classes.items():
            # Choose a random allowed timeslot start or fallback to any slot
            # Note: start_slot must be in [0, self.ds.total_slots - cl.length]
            max_start = max(0, self.ds.total_slots - cl.length)
            
            # Pure GA: generate completely random slot, but if allowed times are specified,
            # we can choose randomly from the start slots of the allowed times as a smart initialisation,
            # or completely random from the whole search space. Let's do completely random to satisfy "pure GA".
            start_slot = random.randint(0, max_start)
            
            # Choose a random allowed room or fallback to any parsed room
            if cl.allowed_rooms:
                room_id = random.choice(cl.allowed_rooms)
            else:
                room_id = random.choice(self.ds.room_ids)
                
            chromosome[class_id] = {
                "start_slot": start_slot,
                "room_id": room_id
            }
        return chromosome

    def initialize_population(self) -> List[Dict[int, Dict[str, int]]]:
        self.population = [
            self.generate_random_chromosome()
            for _ in range(self.population_size)
        ]
        logger.info("Initialized fully random population of size %s", self.population_size)
        return self.population

    def uniform_crossover(
        self,
        parent1: Dict[int, Dict[str, int]],
        parent2: Dict[int, Dict[str, int]]
    ) -> Tuple[Dict[int, Dict[str, int]], Dict[int, Dict[str, int]]]:
        child1 = {}
        child2 = {}
        for class_id in self.ds.class_ids:
            if random.random() < 0.5:
                child1[class_id] = parent1[class_id].copy()
                child2[class_id] = parent2[class_id].copy()
            else:
                child1[class_id] = parent2[class_id].copy()
                child2[class_id] = parent1[class_id].copy()
        return child1, child2

    def single_point_crossover(
        self,
        parent1: Dict[int, Dict[str, int]],
        parent2: Dict[int, Dict[str, int]]
    ) -> Tuple[Dict[int, Dict[str, int]], Dict[int, Dict[str, int]]]:
        if len(self.ds.class_ids) < 2:
            return {cid: gene.copy() for cid, gene in parent1.items()}, {cid: gene.copy() for cid, gene in parent2.items()}
            
        point = random.randint(1, len(self.ds.class_ids) - 1)
        child1 = {}
        child2 = {}
        
        for index, class_id in enumerate(self.ds.class_ids):
            if index < point:
                child1[class_id] = parent1[class_id].copy()
                child2[class_id] = parent2[class_id].copy()
            else:
                child1[class_id] = parent2[class_id].copy()
                child2[class_id] = parent1[class_id].copy()
                
        return child1, child2

    def crossover(
        self,
        parent1: Dict[int, Dict[str, int]],
        parent2: Dict[int, Dict[str, int]]
    ) -> Tuple[Dict[int, Dict[str, int]], Dict[int, Dict[str, int]]]:
        if random.random() > self.crossover_rate:
            return {cid: gene.copy() for cid, gene in parent1.items()}, {cid: gene.copy() for cid, gene in parent2.items()}
            
        if random.random() < 0.5:
            return self.uniform_crossover(parent1, parent2)
        return self.single_point_crossover(parent1, parent2)

    def mutate(self, chromosome: Dict[int, Dict[str, int]]) -> Dict[int, Dict[str, int]]:
        mutated = {cid: gene.copy() for cid, gene in chromosome.items()}
        
        for class_id in self.ds.class_ids:
            if random.random() > self.mutation_rate:
                continue
                
            cl = self.ds.classes[class_id]
            mutation_type = random.choice(["slot", "room", "both"])
            
            if mutation_type in {"slot", "both"}:
                max_start = max(0, self.ds.total_slots - cl.length)
                mutated[class_id]["start_slot"] = random.randint(0, max_start)
                
            if mutation_type in {"room", "both"}:
                if cl.allowed_rooms:
                    mutated[class_id]["room_id"] = random.choice(cl.allowed_rooms)
                else:
                    mutated[class_id]["room_id"] = random.choice(self.ds.room_ids)
                    
        return mutated

    def _evaluate(self, chromosome: Dict[int, Dict[str, int]]) -> EvaluatedChromosome:
        fitness, scores = calculate_fitness(chromosome, self.ds)
        return fitness, scores["total_penalty"], scores, chromosome

    def _evaluate_population(self) -> List[EvaluatedChromosome]:
        evaluated = [self._evaluate(chrom) for chrom in self.population]
        # Sort by total penalty (ascending, lower is better)
        evaluated.sort(key=lambda item: item[1])
        return evaluated

    def _tournament_selection(self, evaluated_population: List[EvaluatedChromosome]) -> Dict[int, Dict[str, int]]:
        size = min(self.tournament_size, len(evaluated_population))
        contestants = random.sample(evaluated_population, size)
        contestants.sort(key=lambda item: item[1])
        return {cid: gene.copy() for cid, gene in contestants[0][3].items()}

    def _log_generation(self, generation: int, best: EvaluatedChromosome) -> None:
        fitness, penalty, scores, _ = best
        
        # Log to terminal using the exact 4 custom penalty categories
        logger.info(
            "Generation %s | best_fitness=%.10f | total_penalty=%d | Student=%d | Room=%d | Time=%d | Dist=%d",
            generation,
            fitness,
            penalty,
            scores["student"],
            scores["room"],
            scores["time"],
            scores["distribution"]
        )
        
        self.history.append({
            "generation": generation,
            "fitness": fitness,
            "penalty": penalty,
            "student": scores["student"],
            "room": scores["room"],
            "time": scores["time"],
            "dist": scores["distribution"],
            "scores": {k: v for k, v in scores.items()}  # fast copy
        })

    def run(self, generations: int = 100) -> Dict[int, Dict[str, int]]:
        self.history = []
        if not self.population:
            self.initialize_population()
            
        evaluated = self._evaluate_population()
        best_overall = evaluated[0]
        self._log_generation(0, best_overall)
        
        elite_count = max(1, int(self.population_size * self.elitism_rate))
        
        for generation in range(1, generations + 1):
            # Elitism: carry over the best individuals directly
            next_population = [
                {cid: gene.copy() for cid, gene in item[3].items()}
                for item in evaluated[:elite_count]
            ]
            
            # Selection, Crossover, and Mutation
            while len(next_population) < self.population_size:
                parent1 = self._tournament_selection(evaluated)
                parent2 = self._tournament_selection(evaluated)
                child1, child2 = self.crossover(parent1, parent2)
                
                next_population.append(self.mutate(child1))
                if len(next_population) < self.population_size:
                    next_population.append(self.mutate(child2))
                    
            self.population = next_population
            evaluated = self._evaluate_population()
            
            if evaluated[0][1] < best_overall[1]:
                best_overall = evaluated[0]
                
            self._log_generation(generation, evaluated[0])
            
            # Early stopping if optimal zero-penalty schedule found
            if best_overall[1] == 0:
                logger.info("Stopping early: zero-penalty chromosome found.")
                break
                
        return {cid: gene.copy() for cid, gene in best_overall[3].items()}

    def run_with_scores(self, generations: int = 100) -> EvaluatedChromosome:
        best_chromosome = self.run(generations)
        return self._evaluate(best_chromosome)
