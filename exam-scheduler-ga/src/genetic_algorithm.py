import random
import logging
from typing import List, Tuple
from .model import Dataset
from .fitness import FitnessCalculator
import numpy as np

logger = logging.getLogger(__name__)

class GeneticAlgorithm:
    def __init__(self, dataset: Dataset, population_size: int = 100, mutation_rate: float = 0.01):
        self.ds = dataset
        self.fitness_calc = FitnessCalculator(dataset)
        self.population_size = population_size
        self.mutation_rate = mutation_rate  # Recommended: 0.01 - 0.02 for this chromosome size
        self.population: List[List[Tuple[int, Tuple[int, ...]]]] = []
        
        # Cache large exams indices for quick lookup (MISTA 2013: 21 large exams)
        self.large_exam_ids = getattr(self.ds, 'large_exams_indices', [])
        
    def generate_random_chromosome(self) -> List[Tuple[int, Tuple[int, ...]]]:
        chromosome = [None] * len(self.ds.exams)
        period_exams = {p: [] for p in self.ds.periods.keys()}
        
        # Order exams: Large exams first, then others randomly
        exam_indices = list(range(len(self.ds.exams)))
        random.shuffle(exam_indices)
        large_exams = [idx for idx in exam_indices if idx in self.large_exam_ids]
        other_exams = [idx for idx in exam_indices if idx not in self.large_exam_ids]
        sorted_exams = large_exams + other_exams
        
        for idx in sorted_exams:
            exam = self.ds.exams[idx]
            # 1. ASSIGN PERIOD (With Smart Heuristic Support)
            if exam.idx in self.ds.time_fixed_exams:
                p_id = self.ds.time_fixed_exams[exam.idx]
            else:
                valid_periods = list(exam.available_periods) if exam.available_periods else list(self.ds.periods.keys())
                if exam.idx in self.large_exam_ids:
                    valid_periods = [p for p in valid_periods if p < 24]
                    if not valid_periods:
                        valid_periods = list(exam.available_periods) if exam.available_periods else list(self.ds.periods.keys())
                
                # Greedy Selection: Pick period with minimum conflict among already scheduled exams
                sampled_periods = random.sample(valid_periods, min(len(valid_periods), 10))
                best_periods, min_conf = [], float('inf')
                for p in sampled_periods:
                    conflicts = sum(self.ds.conflict_matrix[exam.idx][other] for other in period_exams[p])
                    if conflicts < min_conf:
                        min_conf = conflicts
                        best_periods = [p]
                    elif conflicts == min_conf:
                        best_periods.append(p)
                
                p_id = random.choice(best_periods)
                
            period_exams[p_id].append(exam.idx)
                
            # 2. ASSIGN ROOMS
            if exam.idx in self.ds.room_fixed_exams:
                rooms = (self.ds.room_fixed_exams[exam.idx],)
            else:
                available = list(exam.available_rooms) if exam.available_rooms else list(self.ds.rooms.keys())
                random.shuffle(available)
                assigned_rooms = []
                capacity = 0
                
                for r_id in available:
                    if capacity >= exam.students_count: break
                    assigned_rooms.append(r_id)
                    room = self.ds.rooms[r_id]
                    capacity += room.alt_size if exam.alt_seating else room.size
                
                if not assigned_rooms and available:
                    assigned_rooms.append(available[0])
                
                rooms = tuple(assigned_rooms)
                
            chromosome[exam.idx] = (p_id, rooms)
            
        return chromosome
        
    def initialize_population(self):
        logger.info(f"Initializing heuristic population of size {self.population_size}")
        for _ in range(self.population_size):
            self.population.append(self.generate_random_chromosome())

    def crossover(self, parent1: List[Tuple[int, Tuple[int, ...]]], parent2: List[Tuple[int, Tuple[int, ...]]]) -> Tuple[List[Tuple[int, Tuple[int, ...]]], List[Tuple[int, Tuple[int, ...]]]]:
        """
        Performs Uniform Crossover. 
        Uniform crossover preserves highly-constrained structures better than Single-Point for timetabling.
        """
        child1 = [None] * len(parent1)
        child2 = [None] * len(parent2)
        
        for idx in range(len(parent1)):
            if random.random() < 0.5:
                child1[idx] = parent1[idx]
                child2[idx] = parent2[idx]
            else:
                child1[idx] = parent2[idx]
                child2[idx] = parent1[idx]
                
        return child1, child2
        
    def mutate(self, chromosome: List[Tuple[int, Tuple[int, ...]]]) -> List[Tuple[int, Tuple[int, ...]]]:
        # 2.223 gen için tek tek random.random() çağırmak yerine
        # Kaç genin mutasyona uğrayacağını baştan simüle edelim
        num_mutations = np.random.binomial(len(chromosome), self.mutation_rate)
        if num_mutations == 0:
            return chromosome

        new_chromosome = chromosome.copy()
        # Mutasyona uğrayacak rastgele indeksleri seçelim
        mutation_indices = random.sample(range(len(chromosome)), num_mutations)
        
        for idx in mutation_indices:
            is_time_fixed = idx in self.ds.time_fixed_exams
            is_room_fixed = idx in self.ds.room_fixed_exams
            
            if is_time_fixed and is_room_fixed:
                continue
                
            p_id, rooms = new_chromosome[idx]
            
            # 1. PERIOD MUTATION
            if not is_time_fixed:
                if idx in self.large_exam_ids:
                    valid_periods = [p for p in range(24) if (not self.ds.exams[idx].available_periods or p in self.ds.exams[idx].available_periods)]
                    if valid_periods: p_id = random.choice(valid_periods)
                else:
                    opts = list(self.ds.exams[idx].available_periods) if self.ds.exams[idx].available_periods else list(self.ds.periods.keys())
                    p_id = random.choice(opts)
                    
            # 2. ROOM MUTATION
            # 2. ROOM MUTATION
        if not is_room_fixed:
            exam = self.ds.exams[idx]
            available = list(exam.available_rooms) if exam.available_rooms else list(self.ds.rooms.keys())
            random.shuffle(available)
            assigned_rooms = []
            capacity = 0
            
            # SİTEDEKİ METİN KURALI: Kapasite dolana kadar akıllıca oda ekle (Room Split desteği)
            for r_id in available:
                if capacity >= exam.students_count: break
                assigned_rooms.append(r_id)
                room = self.ds.rooms[r_id]
                capacity += room.alt_size if exam.alt_seating else room.size
                    
            # Eğer havuzda oda kalmadıysa ama hala kapasite yetmediyse eldekileri kullan
            if not assigned_rooms and available:
                assigned_rooms.append(available[0])
                
            rooms = tuple(assigned_rooms)
                
            new_chromosome[idx] = (p_id, rooms)
            
        return new_chromosome

    def _tournament_selection(self, evaluated_population: List[Tuple[float, List[Tuple[int, Tuple[int, ...]]]]], tournament_size: int = 5) -> List[Tuple[int, Tuple[int, ...]]]:
        """Selects the best individual out of a randomly sampled subset to preserve diversity."""
        tournament = random.sample(evaluated_population, tournament_size)
        tournament.sort(key=lambda x: x[0])  # Sort by penalty ascending
        return tournament[0][1]
        
    def _repair_operator(self, chromosome: List[Tuple[int, Tuple[int, ...]]]) -> List[Tuple[int, Tuple[int, ...]]]:
        """Finds the top 3 exams with the most direct conflicts and shifts them to a better period."""
        repaired = chromosome.copy()
        
        # Calculate direct conflicts per exam
        period_exams = [[] for _ in range(30)]
        for idx, (p_id, _) in enumerate(repaired):
            period_exams[p_id].append(idx)
            
        exam_conflicts = {}
        for p_id in range(1, 30):
            exams = period_exams[p_id]
            for ex1 in exams:
                conflicts = 0
                for ex2 in exams:
                    if ex1 != ex2:
                        conflicts += self.ds.conflict_matrix[ex1][ex2]
                if conflicts > 0:
                    exam_conflicts[ex1] = conflicts
                    
        if not exam_conflicts:
            return repaired
            
        # Get top 20 exams with most conflicts to repair
        sorted_exams = sorted(exam_conflicts.items(), key=lambda x: x[1], reverse=True)
        top_exams = [ex for ex, _ in sorted_exams[:20]]
        
        # Shift them to a period with minimum conflicts
        for ex in top_exams:
            if ex in self.ds.time_fixed_exams:
                continue
                
            best_p = repaired[ex][0]
            min_conflicts = float('inf')
            
            valid_periods = list(self.ds.exams[ex].available_periods) if self.ds.exams[ex].available_periods else list(self.ds.periods.keys())
            if ex in self.large_exam_ids:
                valid_periods = [p for p in valid_periods if p < 24]
                
            if not valid_periods:
                continue
                
            for p_id in valid_periods:
                c = 0
                for other_ex in period_exams[p_id]:
                    if other_ex != ex:
                        c += self.ds.conflict_matrix[ex][other_ex]
                if c < min_conflicts:
                    min_conflicts = c
                    best_p = p_id
                    if c == 0:
                        break  # Found a completely conflict-free period!
                        
            # Update chromosome and period tracking
            old_p = repaired[ex][0]
            if old_p != best_p:
                repaired[ex] = (best_p, repaired[ex][1])
                period_exams[old_p].remove(ex)
                period_exams[best_p].append(ex)
                
        return repaired

    def run(self, generations: int = 100):
        self.initialize_population()
        
        for gen in range(generations):
            # Evaluate fitness across current population
            evaluated = []
            for chrom in self.population:
                fitness, _ = self.fitness_calc.calculate_fitness(chrom)
                evaluated.append((fitness, chrom))
                
            # Sort population to identify top candidates
            evaluated.sort(key=lambda x: x[0])
            
            # Apply Repair Operator to the best chromosome to fix severe conflicts
            best_chrom = evaluated[0][1]
            repaired_chrom = self._repair_operator(best_chrom)
            rep_fitness, _ = self.fitness_calc.calculate_fitness(repaired_chrom)
            if rep_fitness < evaluated[0][0]:
                evaluated[0] = (rep_fitness, repaired_chrom)
                
            # Re-sort just in case the repair improved the best beyond others
            evaluated.sort(key=lambda x: x[0])

            best_fitness = evaluated[0][0]
            logger.info(f"Generation {gen}: Best Fitness (Penalty) = {best_fitness}")
            
            # Elitism: Keep the top 5% of chromosomes unconditionally to guarantee progress
            elitism_count = max(2, int(self.population_size * 0.05))
            next_population = [chrom for _, chrom in evaluated[:elitism_count]]
            
            # Reproduction Loop
            while len(next_population) < self.population_size:
                # Use tournament selection instead of simple slice to prevent premature convergence
                parent1 = self._tournament_selection(evaluated, tournament_size=5)
                parent2 = self._tournament_selection(evaluated, tournament_size=5)
                
                # Apply Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Apply Controlled Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                next_population.append(child1)
                if len(next_population) < self.population_size:
                    next_population.append(child2)
                    
            self.population = next_population
            
        # Re-evaluate the final population to return the absolute best item
        evaluated = []
        for chrom in self.population:
            fitness, _ = self.fitness_calc.calculate_fitness(chrom)
            evaluated.append((fitness, chrom))
        evaluated.sort(key=lambda x: x[0])
        
        return evaluated[0][1]