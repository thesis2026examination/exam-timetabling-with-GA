import copy
import logging
import random
from typing import Dict, List, Tuple

from .fitness import Chromosome, FitnessCalculator
from .model import CSVDataset

logger = logging.getLogger(__name__)

EvaluatedChromosome = Tuple[float, int, Dict[str, int], Chromosome]


class GeneticAlgorithm:
    def __init__(
        self,
        dataset: CSVDataset,
        population_size: int = 100,
        mutation_rate: float = 0.10,
        crossover_rate: float = 0.90,
        elitism_rate: float = 0.05,
        tournament_size: int = 5,
        max_rooms_per_exam: int = 15,
    ):
        self.ds = dataset
        self.fitness_calc = FitnessCalculator(dataset)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.max_rooms_per_exam = min(max_rooms_per_exam, len(self.ds.classrooms))

        self.timeslot_ids = sorted(self.ds.timeslots)
        self.classroom_ids = sorted(self.ds.classrooms)
        self.population: List[Chromosome] = []
        self.history: List[Dict] = []

    def _random_rooms(self) -> List[int]:
        room_count = random.randint(1, self.max_rooms_per_exam)
        return random.sample(self.classroom_ids, room_count)

    def generate_random_chromosome(self) -> Chromosome:
        chromosome = {}
        for course_id in self.ds.course_ids:
            chromosome[course_id] = {
                "timeslot": random.choice(self.timeslot_ids),
                "rooms": self._random_rooms(),
            }
        return chromosome

    def initialize_population(self) -> List[Chromosome]:
        self.population = [
            self.generate_random_chromosome()
            for _ in range(self.population_size)
        ]
        logger.info("Initialized fully random population of size %s", self.population_size)
        return self.population

    def uniform_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        child1 = {}
        child2 = {}

        for course_id in self.ds.course_ids:
            if random.random() < 0.5:
                child1[course_id] = copy.deepcopy(parent1[course_id])
                child2[course_id] = copy.deepcopy(parent2[course_id])
            else:
                child1[course_id] = copy.deepcopy(parent2[course_id])
                child2[course_id] = copy.deepcopy(parent1[course_id])

        return child1, child2

    def single_point_crossover(
        self,
        parent1: Chromosome,
        parent2: Chromosome,
    ) -> Tuple[Chromosome, Chromosome]:
        if len(self.ds.course_ids) < 2:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        point = random.randint(1, len(self.ds.course_ids) - 1)
        child1 = {}
        child2 = {}

        for index, course_id in enumerate(self.ds.course_ids):
            if index < point:
                child1[course_id] = copy.deepcopy(parent1[course_id])
                child2[course_id] = copy.deepcopy(parent2[course_id])
            else:
                child1[course_id] = copy.deepcopy(parent2[course_id])
                child2[course_id] = copy.deepcopy(parent1[course_id])

        return child1, child2

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        if random.random() < 0.5:
            return self.uniform_crossover(parent1, parent2)
        return self.single_point_crossover(parent1, parent2)

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        mutated = copy.deepcopy(chromosome)

        for course_id in self.ds.course_ids:
            if random.random() > self.mutation_rate:
                continue

            mutation_type = random.choice(["timeslot", "rooms", "both"])
            if mutation_type in {"timeslot", "both"}:
                mutated[course_id]["timeslot"] = random.choice(self.timeslot_ids)

            if mutation_type in {"rooms", "both"}:
                mutated[course_id]["rooms"] = self._random_rooms()

        return mutated

    def _evaluate(self, chromosome: Chromosome) -> EvaluatedChromosome:
        fitness, scores = self.fitness_calc.calculate_fitness(chromosome)
        return fitness, scores["total_penalty"], scores, chromosome

    def _evaluate_population(self) -> List[EvaluatedChromosome]:
        evaluated = [self._evaluate(chromosome) for chromosome in self.population]
        evaluated.sort(key=lambda item: item[1])
        return evaluated

    def _tournament_selection(self, evaluated_population: List[EvaluatedChromosome]) -> Chromosome:
        size = min(self.tournament_size, len(evaluated_population))
        contestants = random.sample(evaluated_population, size)
        contestants.sort(key=lambda item: item[1])
        return copy.deepcopy(contestants[0][3])

    def _log_generation(self, generation: int, best: EvaluatedChromosome) -> None:
        fitness, penalty, scores, _ = best
        hard_penalty = (
            scores["instructor_conflict"]
            + scores["room_conflict"]
            + scores["student_conflict"]
            + scores["capacity_shortage"]
        )
        soft_penalty = penalty - hard_penalty

        logger.info(
            "Generation %s | best_fitness=%.10f | total_penalty=%s | hard=%s | soft=%s | "
            "H1=%s H2=%s H3=%s H4=%s | S1=%s S2_day=%s S2_adj=%s",
            generation,
            fitness,
            penalty,
            hard_penalty,
            soft_penalty,
            scores["instructor_conflict"],
            scores["room_conflict"],
            scores["student_conflict"],
            scores["capacity_shortage"],
            scores["building_spread"],
            scores["same_day_extra_exam"],
            scores["back_to_back_exam"],
        )

        self.history.append({
            "generation": generation,
            "fitness": fitness,
            "penalty": penalty,
            "hard": hard_penalty,
            "soft": soft_penalty,
            "scores": copy.deepcopy(scores)
        })

    def run(self, generations: int = 100) -> Chromosome:
        self.history = []
        if not self.population:
            self.initialize_population()

        evaluated = self._evaluate_population()
        best_overall = evaluated[0]
        self._log_generation(0, best_overall)

        elite_count = max(1, int(self.population_size * self.elitism_rate))

        for generation in range(1, generations + 1):
            next_population = [
                copy.deepcopy(item[3])
                for item in evaluated[:elite_count]
            ]

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

            if best_overall[1] == 0:
                logger.info("Stopping early: zero-penalty chromosome found.")
                break

        return copy.deepcopy(best_overall[3])

    def run_with_scores(self, generations: int = 100) -> EvaluatedChromosome:
        best_chromosome = self.run(generations)
        return self._evaluate(best_chromosome)
