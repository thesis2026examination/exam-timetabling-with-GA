import random
import logging
from typing import List, Tuple
from .model import Dataset
from .fitness import FitnessCalculator
import numpy as np

logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    def __init__(self, dataset: Dataset, population_size: int = 100, mutation_rate: float = 0.015):
        self.ds = dataset
        self.fitness_calc = FitnessCalculator(dataset)
        self.population_size = population_size

        # Keep a conservative base mutation rate. Very high external values tend
        # to destroy useful schemas, so they are clamped to the original default.
        self.base_mutation_rate = mutation_rate if mutation_rate < 0.05 else 0.015
        self.mutation_rate = self.base_mutation_rate

        self.population: List[List[Tuple[int, Tuple[int, ...]]]] = []

        # Cached structures used by DSatur initialization, mutation, and repair.
        self.large_exam_ids = getattr(self.ds, 'large_exams_indices', [])
        self.large_exam_set = set(self.large_exam_ids)
        self.use_dsatur_initialization = True
        self.exam_conflict_degree = np.asarray(self.ds.conflict_matrix.sum(axis=1)).ravel()
        self.b2b_neighbors = self._build_back_to_back_neighbors()

    def _build_back_to_back_neighbors(self):
        neighbors = {p: [] for p in self.ds.periods.keys()}
        period_ids = sorted(self.ds.periods.keys())
        for p1, p2 in zip(period_ids, period_ids[1:]):
            if self.ds.periods[p1].day == self.ds.periods[p2].day:
                neighbors[p1].append(p2)
                neighbors[p2].append(p1)
        return neighbors

    def _valid_periods_for_exam(self, idx: int) -> List[int]:
        exam = self.ds.exams[idx]
        valid_periods = list(exam.available_periods) if exam.available_periods else list(self.ds.periods.keys())

        if idx in self.large_exam_set:
            preferred = [p for p in valid_periods if p < 24]
            if preferred:
                valid_periods = preferred

        return valid_periods

    def _can_exam_use_period(self, idx: int, period_id: int) -> bool:
        # Used by swap mutation. A swap must not quietly violate time-fixed,
        # exam availability, or the large-exam period restriction.
        if idx in self.ds.time_fixed_exams:
            return self.ds.time_fixed_exams[idx] == period_id
        return period_id in self._valid_periods_for_exam(idx)

    def _assign_rooms_for_exam(self, idx: int) -> Tuple[int, ...]:
        exam = self.ds.exams[idx]
        if idx in self.ds.room_fixed_exams:
            return (self.ds.room_fixed_exams[idx],)

        available = list(exam.available_rooms) if exam.available_rooms else list(self.ds.rooms.keys())
        random.shuffle(available)
        assigned_rooms = []
        capacity = 0

        for r_id in available:
            if capacity >= exam.students_count:
                break
            assigned_rooms.append(r_id)
            room = self.ds.rooms[r_id]
            capacity += room.alt_size if exam.alt_seating else room.size

        if not assigned_rooms and available:
            assigned_rooms.append(available[0])

        return tuple(assigned_rooms)

    def _choose_period_by_local_cost(self, idx: int, valid_periods: List[int], period_exams) -> int:
        # Lightweight local scoring: direct conflicts dominate; back-to-back
        # conflicts and period penalty are tie-breakers. This keeps initialization
        # and mutation scalable without calling full fitness.
        scored_periods = []
        for p_id in valid_periods:
            same_period = period_exams[p_id]
            direct_conflicts = int(self.ds.conflict_matrix[idx, same_period].sum()) if same_period else 0

            b2b_conflicts = 0
            for neighbor in self.b2b_neighbors.get(p_id, []):
                neighbor_exams = period_exams[neighbor]
                if neighbor_exams:
                    b2b_conflicts += int(self.ds.conflict_matrix[idx, neighbor_exams].sum())

            period_penalty = self.ds.periods[p_id].penalty
            scored_periods.append((direct_conflicts, b2b_conflicts, period_penalty, p_id))

        scored_periods.sort(key=lambda item: item[:3])
        best_score = scored_periods[0][:3]
        equally_good = [p_id for dc, b2b, penalty, p_id in scored_periods if (dc, b2b, penalty) == best_score]
        return random.choice(equally_good)

    def generate_dsatur_chromosome(self) -> List[Tuple[int, Tuple[int, ...]]]:
        """
        DSatur-inspired graph-coloring initialization.

        The next exam is chosen by highest saturation degree: how many distinct
        periods are already used by its scheduled conflicting neighbors. Ties
        favor weighted conflict degree, larger exams, then fewer valid periods.
        """
        chromosome = [None] * len(self.ds.exams)
        period_exams = {p: [] for p in self.ds.periods.keys()}
        unscheduled = set(range(len(self.ds.exams)))
        neighbor_periods = [set() for _ in self.ds.exams]

        # Anchor time-fixed exams first because they constrain the graph.
        for idx, p_id in self.ds.time_fixed_exams.items():
            chromosome[idx] = (p_id, self._assign_rooms_for_exam(idx))
            period_exams[p_id].append(idx)
            unscheduled.discard(idx)

        for fixed_idx, gene in enumerate(chromosome):
            if gene is None:
                continue
            p_id, _ = gene
            neighbors = np.flatnonzero(self.ds.conflict_matrix[fixed_idx] > 0)
            for n_idx in neighbors:
                if n_idx in unscheduled:
                    neighbor_periods[n_idx].add(p_id)

        while unscheduled:
            candidates = list(unscheduled)
            max_saturation = max(len(neighbor_periods[idx]) for idx in candidates)
            candidates = [idx for idx in candidates if len(neighbor_periods[idx]) == max_saturation]

            max_degree = max(self.exam_conflict_degree[idx] for idx in candidates)
            candidates = [idx for idx in candidates if self.exam_conflict_degree[idx] == max_degree]

            max_students = max(self.ds.exams[idx].students_count for idx in candidates)
            candidates = [idx for idx in candidates if self.ds.exams[idx].students_count == max_students]

            min_periods = min(len(self._valid_periods_for_exam(idx)) for idx in candidates)
            candidates = [idx for idx in candidates if len(self._valid_periods_for_exam(idx)) == min_periods]

            idx = random.choice(candidates)
            valid_periods = self._valid_periods_for_exam(idx)
            p_id = self._choose_period_by_local_cost(idx, valid_periods, period_exams)

            chromosome[idx] = (p_id, self._assign_rooms_for_exam(idx))
            period_exams[p_id].append(idx)
            unscheduled.remove(idx)

            neighbors = np.flatnonzero(self.ds.conflict_matrix[idx] > 0)
            for n_idx in neighbors:
                if n_idx in unscheduled:
                    neighbor_periods[n_idx].add(p_id)

        return chromosome

    def generate_random_chromosome(self) -> List[Tuple[int, Tuple[int, ...]]]:
        if self.use_dsatur_initialization:
            return self.generate_dsatur_chromosome()

        chromosome = [None] * len(self.ds.exams)
        period_exams = {p: [] for p in self.ds.periods.keys()}

        # Fallback to the original large-exam-first heuristic if DSatur is disabled.
        exam_indices = list(range(len(self.ds.exams)))
        random.shuffle(exam_indices)
        large_exams = [idx for idx in exam_indices if idx in self.large_exam_set]
        other_exams = [idx for idx in exam_indices if idx not in self.large_exam_set]
        sorted_exams = large_exams + other_exams

        for idx in sorted_exams:
            exam = self.ds.exams[idx]
            if exam.idx in self.ds.time_fixed_exams:
                p_id = self.ds.time_fixed_exams[exam.idx]
            else:
                valid_periods = self._valid_periods_for_exam(exam.idx)
                p_id = self._choose_period_by_local_cost(exam.idx, valid_periods, period_exams)

            period_exams[p_id].append(exam.idx)
            chromosome[exam.idx] = (p_id, self._assign_rooms_for_exam(exam.idx))

        return chromosome

    def initialize_population(self):
        logger.info(
            "Initializing population of size %s using %s initialization",
            self.population_size,
            "DSatur-inspired" if self.use_dsatur_initialization else "large-exam-first heuristic",
        )
        for _ in range(self.population_size):
            self.population.append(self.generate_random_chromosome())

    def _period_exam_lists(self, chromosome):
        period_exams = [[] for _ in range(30)]
        for idx, (p_id, _) in enumerate(chromosome):
            period_exams[p_id].append(idx)
        return period_exams

    def _local_direct_conflict(self, idx: int, period_id: int, period_exams: List[List[int]]) -> int:
        others = [ex for ex in period_exams[period_id] if ex != idx]
        return int(self.ds.conflict_matrix[idx, others].sum()) if others else 0

    def _make_conservative_child(self, base_parent, donor_parent):
        """
        Preserve one parent's full timetable structure, then import donor genes
        only when the move is locally safe for direct conflicts.

        This avoids the common timetabling crossover failure where uniform gene
        mixing creates period groups that neither parent ever had.
        """
        child = base_parent.copy()
        period_exams = self._period_exam_lists(child)
        indices = list(range(len(child)))
        random.shuffle(indices)

        for idx in indices:
            if idx in self.ds.time_fixed_exams:
                continue

            current_period = child[idx][0]
            donor_period, donor_rooms = donor_parent[idx]
            if donor_period == current_period:
                continue
            if not self._can_exam_use_period(idx, donor_period):
                continue

            current_conflict = self._local_direct_conflict(idx, current_period, period_exams)

            if idx in period_exams[current_period]:
                period_exams[current_period].remove(idx)
            donor_conflict = self._local_direct_conflict(idx, donor_period, period_exams)

            # Accept improvements, and accept a few equal moves to keep diversity.
            if donor_conflict < current_conflict or (donor_conflict == current_conflict and random.random() < 0.10):
                child[idx] = (donor_period, donor_rooms)
                period_exams[donor_period].append(idx)
            else:
                period_exams[current_period].append(idx)

        return child

    def crossover(self, parent1: List[Tuple[int, Tuple[int, ...]]], parent2: List[Tuple[int, Tuple[int, ...]]]) -> Tuple[List[Tuple[int, Tuple[int, ...]]], List[Tuple[int, Tuple[int, ...]]]]:
        """
        Conservative conflict-aware crossover.

        Each child starts as a full parent timetable. Genes from the other parent
        are imported only when the local direct-conflict delta is non-worsening.
        """
        child1 = self._make_conservative_child(parent1, parent2)
        child2 = self._make_conservative_child(parent2, parent1)
        return child1, child2

    def _find_problematic_exams(self, period_exams: List[List[int]]) -> List[int]:
        """
        Return exams involved in direct or back-to-back conflicts.

        Direct conflicts receive a larger score because reducing them is the main
        goal and their fitness weight is much higher.
        """
        problem_scores = {}

        for p_id in range(1, 30):
            exams = period_exams[p_id]
            if len(exams) < 2:
                continue
            sub = self.ds.conflict_matrix[np.ix_(exams, exams)]
            row_sums = sub.sum(axis=1)
            for i, ex in enumerate(exams):
                if row_sums[i] > 0:
                    problem_scores[ex] = problem_scores.get(ex, 0) + int(row_sums[i]) * 10

        for p_id, neighbors in self.b2b_neighbors.items():
            exams = period_exams[p_id]
            if not exams:
                continue
            for neighbor in neighbors:
                if neighbor < p_id:
                    continue
                neighbor_exams = period_exams[neighbor]
                if not neighbor_exams:
                    continue
                c_mat = self.ds.conflict_matrix[np.ix_(exams, neighbor_exams)]
                if not c_mat.any():
                    continue

                left_scores = c_mat.sum(axis=1)
                right_scores = c_mat.sum(axis=0)
                for i, ex in enumerate(exams):
                    if left_scores[i] > 0:
                        problem_scores[ex] = problem_scores.get(ex, 0) + int(left_scores[i])
                for j, ex in enumerate(neighbor_exams):
                    if right_scores[j] > 0:
                        problem_scores[ex] = problem_scores.get(ex, 0) + int(right_scores[j])

        return [ex for ex, _ in sorted(problem_scores.items(), key=lambda item: item[1], reverse=True)]

    def _select_mutation_indices(self, num_mutations: int, problematic: List[int], chromosome_len: int) -> List[int]:
        selected = set()
        target_problematic = int(round(num_mutations * 0.70))
        available_problematic = [idx for idx in problematic if idx not in self.ds.time_fixed_exams]

        if available_problematic:
            selected.update(random.sample(available_problematic, min(target_problematic, len(available_problematic))))

        while len(selected) < num_mutations:
            selected.add(random.randrange(chromosome_len))

        return list(selected)

    def mutate(self, chromosome: List[Tuple[int, Tuple[int, ...]]]) -> List[Tuple[int, Tuple[int, ...]]]:
        """
        Constraint-aware mutation.

        Most mutation pressure is placed on exams currently involved in direct or
        back-to-back conflicts, while a smaller random component preserves diversity.
        No full fitness calls are made here.
        """
        num_mutations = np.random.binomial(len(chromosome), self.mutation_rate)
        num_mutations = min(num_mutations, 12)
        if num_mutations == 0:
            return chromosome

        new_chromosome = chromosome.copy()

        period_exams = [[] for _ in range(30)]
        for i, (p_id, _) in enumerate(new_chromosome):
            period_exams[p_id].append(i)

        problematic = self._find_problematic_exams(period_exams)
        mutation_indices = self._select_mutation_indices(num_mutations, problematic, len(chromosome))

        for idx in mutation_indices:
            is_time_fixed = idx in self.ds.time_fixed_exams
            is_room_fixed = idx in self.ds.room_fixed_exams

            if is_time_fixed and is_room_fixed:
                continue

            # Keep the existing swap mutation, but only for non-fixed exams.
            if not is_time_fixed and random.random() < 0.30:
                swap_with = random.randint(0, len(new_chromosome) - 1)
                if swap_with != idx and swap_with not in self.ds.time_fixed_exams:
                    p_id_1, rooms_1 = new_chromosome[idx]
                    p_id_2, rooms_2 = new_chromosome[swap_with]

                    if not self._can_exam_use_period(idx, p_id_2):
                        swap_with = None
                    elif not self._can_exam_use_period(swap_with, p_id_1):
                        swap_with = None

                if swap_with is not None and swap_with != idx and swap_with not in self.ds.time_fixed_exams:
                    p_id_1, rooms_1 = new_chromosome[idx]
                    p_id_2, rooms_2 = new_chromosome[swap_with]

                    new_chromosome[idx] = (p_id_2, rooms_1)
                    new_chromosome[swap_with] = (p_id_1, rooms_2)

                    if idx in period_exams[p_id_1]:
                        period_exams[p_id_1].remove(idx)
                    period_exams[p_id_2].append(idx)
                    if swap_with in period_exams[p_id_2]:
                        period_exams[p_id_2].remove(swap_with)
                    period_exams[p_id_1].append(swap_with)
                    continue

            p_id, rooms = new_chromosome[idx]

            if not is_time_fixed:
                valid_periods = self._valid_periods_for_exam(idx)

                if valid_periods:
                    if idx in period_exams[p_id]:
                        period_exams[p_id].remove(idx)

                    p_id = self._choose_period_by_local_cost(idx, valid_periods, period_exams)
                    period_exams[p_id].append(idx)

            if not is_room_fixed:
                rooms = self._assign_rooms_for_exam(idx)

            new_chromosome[idx] = (p_id, rooms)

        return new_chromosome

    def _tournament_selection(self, evaluated_population: List[Tuple[float, List[Tuple[int, Tuple[int, ...]]]]], tournament_size: int = 7) -> List[Tuple[int, Tuple[int, ...]]]:
        """Tournament selection with the existing selection pressure."""
        tournament = random.sample(evaluated_population, tournament_size)
        tournament.sort(key=lambda x: x[0])
        return tournament[0][1]

    def _repair_operator(self, chromosome: List[Tuple[int, Tuple[int, ...]]]) -> Tuple[List[Tuple[int, Tuple[int, ...]]], int]:
        """
        Lightweight direct-conflict repair.

        Only the worst direct-conflict exams are considered. A move is accepted
        only when the local direct-conflict count strictly decreases.
        """
        repaired = chromosome.copy()
        period_exams = [[] for _ in range(30)]
        for idx, (p_id, _) in enumerate(repaired):
            period_exams[p_id].append(idx)

        exam_conflicts = {}
        for p_id in range(1, 30):
            exams = period_exams[p_id]
            if len(exams) < 2:
                continue
            sub = self.ds.conflict_matrix[np.ix_(exams, exams)]
            row_sums = sub.sum(axis=1)
            for i, ex in enumerate(exams):
                if row_sums[i] > 0:
                    exam_conflicts[ex] = int(row_sums[i])

        if not exam_conflicts:
            return repaired, 0

        sorted_exams = sorted(exam_conflicts.items(), key=lambda x: x[1], reverse=True)
        top_exams = [ex for ex, _ in sorted_exams[:60]]
        accepted_repairs = 0

        for ex in top_exams:
            if ex in self.ds.time_fixed_exams:
                continue

            valid_periods = self._valid_periods_for_exam(ex)
            if not valid_periods:
                continue

            old_p = repaired[ex][0]
            if ex in period_exams[old_p]:
                period_exams[old_p].remove(ex)

            old_conflict = int(self.ds.conflict_matrix[ex, period_exams[old_p]].sum()) if period_exams[old_p] else 0
            best_p = old_p
            best_conflict = old_conflict

            for p_id in valid_periods:
                if p_id == old_p:
                    continue
                others = period_exams[p_id]
                new_conflict = int(self.ds.conflict_matrix[ex, others].sum()) if others else 0
                if new_conflict < best_conflict:
                    best_conflict = new_conflict
                    best_p = p_id
                    if new_conflict == 0:
                        break

            if best_p != old_p and best_conflict < old_conflict:
                repaired[ex] = (best_p, repaired[ex][1])
                accepted_repairs += 1

            period_exams[best_p].append(ex)

        return repaired, accepted_repairs

    def run(self, generations: int = 100):
        self.initialize_population()

        patience = 4
        stagnation = 0
        last_best = float('inf')

        for gen in range(generations):
            evaluated = []
            for chrom in self.population:
                fitness, _ = self.fitness_calc.calculate_fitness(chrom)
                evaluated.append((fitness, chrom))

            evaluated.sort(key=lambda x: x[0])
            avg_fitness = sum(fitness for fitness, _ in evaluated) / len(evaluated)

            # Repair only a small elite slice. The repair itself uses local direct
            # conflict deltas; the full fitness call here validates whole-solution
            # improvement without placing full fitness inside the repair loop.
            repair_count = max(1, int(self.population_size * 0.12))
            accepted_repairs_total = 0
            for i in range(repair_count):
                repaired, accepted_repairs = self._repair_operator(evaluated[i][1])
                accepted_repairs_total += accepted_repairs
                rep_fitness, _ = self.fitness_calc.calculate_fitness(repaired)
                if rep_fitness < evaluated[i][0]:
                    evaluated[i] = (rep_fitness, repaired)

            evaluated.sort(key=lambda x: x[0])
            best_fitness = evaluated[0][0]
            avg_fitness_after_repair = sum(fitness for fitness, _ in evaluated) / len(evaluated)

            if best_fitness < last_best:
                stagnation = 0
                self.mutation_rate = self.base_mutation_rate
            else:
                stagnation += 1

            cataclysm_triggered = False

            if stagnation >= patience * 4:
                logger.warning(
                    "Generation %s: [CATACLYSM] no improvement for %s generations; refreshing population.",
                    gen,
                    stagnation,
                )
                self.mutation_rate = self.base_mutation_rate
                stagnation = 0
                cataclysm_triggered = True
            elif stagnation >= patience:
                self.mutation_rate = min(0.08, self.base_mutation_rate + (stagnation - patience) * 0.005)
                logger.info(
                    "Generation %s: [ADAPTIVE] stagnation=%s, mutation_rate=%.4f",
                    gen,
                    stagnation,
                    self.mutation_rate,
                )
            else:
                self.mutation_rate = self.base_mutation_rate

            last_best = best_fitness

            logger.info(
                "Generation %s: Best Fitness = %.2f | Avg Fitness = %.2f (pre-repair %.2f) | "
                "accepted_repairs=%s | mut_rate=%.4f | stagnation=%s/%s",
                gen,
                best_fitness,
                avg_fitness_after_repair,
                avg_fitness,
                accepted_repairs_total,
                self.mutation_rate,
                stagnation,
                patience,
            )

            if cataclysm_triggered:
                elitism_count = 1
            else:
                elitism_count = max(4, int(self.population_size * 0.10))

            next_population = [chrom for _, chrom in evaluated[:elitism_count]]

            if cataclysm_triggered:
                fresh_count = int(self.population_size * 0.60)
                for _ in range(fresh_count):
                    next_population.append(self.generate_random_chromosome())

            while len(next_population) < self.population_size:
                parent1 = self._tournament_selection(evaluated, tournament_size=7)
                parent2 = self._tournament_selection(evaluated, tournament_size=7)

                child1, child2 = self.crossover(parent1, parent2)

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                next_population.append(child1)
                if len(next_population) < self.population_size:
                    next_population.append(child2)

            self.population = next_population

        evaluated = []
        for chrom in self.population:
            fitness, _ = self.fitness_calc.calculate_fitness(chrom)
            evaluated.append((fitness, chrom))
        evaluated.sort(key=lambda x: x[0])

        return evaluated[0][1]
