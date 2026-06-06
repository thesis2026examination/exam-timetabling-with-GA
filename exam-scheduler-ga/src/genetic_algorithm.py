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
        elitism_rate: float = 0.10,
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
        
        # Precompute lookups for student conflicts to avoid scanning dataset lists in local search/heuristic
        self.conflict_dict = {}
        self.class_conflict_partners = {}
        for c1, c2, count in self.ds.conflict_pairs:
            self.conflict_dict[(c1, c2)] = count
            self.conflict_dict[(c2, c1)] = count
            self.class_conflict_partners.setdefault(c1, []).append((c2, count))
            self.class_conflict_partners.setdefault(c2, []).append((c1, count))
        
    def generate_random_chromosome(self) -> Dict[int, Dict[str, int]]:
        chromosome = {}
        for class_id, cl in self.ds.classes.items():
            # Choose a random allowed timeslot start or fallback to any slot
            # Note: start_slot must be in [0, self.ds.total_slots - cl.length]
            max_start = max(0, self.ds.total_slots - cl.length)
            start_slot = random.randint(0, max_start)
            
            # Capacity-aware room assignment
            allowed_rooms = cl.allowed_rooms if cl.allowed_rooms else self.ds.room_ids
            enrollment = self.ds.class_enrollments[class_id]
            fit_rooms = [r_id for r_id in allowed_rooms if self.ds.room_capacities[r_id] >= enrollment]
            
            if fit_rooms:
                room_id = random.choice(fit_rooms)
            else:
                # Fallback to the largest room in allowed_rooms to minimize capacity penalty
                room_id = max(allowed_rooms, key=lambda r_id: self.ds.room_capacities[r_id])
                
            chromosome[class_id] = {
                "start_slot": start_slot,
                "room_id": room_id
            }
        return chromosome

    def generate_heuristic_chromosome(self) -> Dict[int, Dict[str, int]]:
        chromosome = {}
        # Sort classes by enrollment size (largest first)
        sorted_classes = sorted(self.ds.class_ids, key=lambda cid: self.ds.class_enrollments[cid], reverse=True)
        
        # Track room assignments: room_id -> list of (start_slot, end_slot)
        room_assignments = {}
        
        for class_id in sorted_classes:
            cl = self.ds.classes[class_id]
            max_start = max(0, self.ds.total_slots - cl.length)
            
            best_slot = None
            best_room = None
            best_penalty = float('inf')
            
            # Determine candidates (with capacity-aware room filtering)
            candidates = []
            allowed_rooms = cl.allowed_rooms if cl.allowed_rooms else self.ds.room_ids
            enrollment = self.ds.class_enrollments[class_id]
            fit_rooms = [r_id for r_id in allowed_rooms if self.ds.room_capacities[r_id] >= enrollment]
            rooms_to_try = fit_rooms if fit_rooms else allowed_rooms
            
            if cl.allowed_times:
                allowed_slots_list = list(self.ds.allowed_slots[class_id]) if self.ds.allowed_slots[class_id] else []
                if allowed_slots_list:
                    # Sample up to 3 candidates
                    for _ in range(min(3, len(allowed_slots_list))):
                        slot = random.choice(allowed_slots_list)
                        room = random.choice(rooms_to_try)
                        candidates.append((slot, room))
            
            # If no candidates from allowed slots, sample randomly from all start slots
            if not candidates:
                for _ in range(3):
                    slot = random.randint(0, max_start)
                    room = random.choice(rooms_to_try)
                    candidates.append((slot, room))
            
            # Find the best candidate that minimizes immediate conflict with already scheduled classes
            for slot, room in candidates:
                penalty = 0.0
                cl_end = slot + cl.length
                
                # Check student conflicts with already placed classes
                partners = self.class_conflict_partners.get(class_id, [])
                for other_cid, shared_count in partners:
                    if other_cid in chromosome:
                        other_slot = chromosome[other_cid]["start_slot"]
                        other_end = other_slot + self.ds.class_lengths[other_cid]
                        if slot < other_end and other_slot < cl_end:
                            penalty += shared_count * 5.0 # student conflict weight
                            
                # Check room overlaps with already placed classes using room_assignments tracker
                room_classes = room_assignments.get(room, [])
                for other_start, other_end in room_classes:
                    if slot < other_end and other_start < cl_end:
                        penalty += 500.0 # High penalty for room overlap
                        break
                            
                # Capacity penalty
                capacity_violation = max(0, self.ds.class_enrollments[class_id] - self.ds.room_capacities[room])
                penalty += capacity_violation * 1.0 # room violation weight
                
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_slot = slot
                    best_room = room
                    
            chromosome[class_id] = {
                "start_slot": best_slot,
                "room_id": best_room
            }
            # Record assignment to avoid O(N) scanning
            room_assignments.setdefault(best_room, []).append((best_slot, best_slot + cl.length))
            
        return chromosome

    def initialize_population(self) -> List[Dict[int, Dict[str, int]]]:
        # Mix 50% heuristic initialization with 50% random to maintain diversity
        heuristic_count = self.population_size // 2
        random_count = self.population_size - heuristic_count
        
        self.population = []
        for _ in range(heuristic_count):
            self.population.append(self.generate_heuristic_chromosome())
        for _ in range(random_count):
            self.population.append(self.generate_random_chromosome())
            
        logger.info("Initialized population of size %s (50%% heuristic, 50%% random)", self.population_size)
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

    def _evaluate_local_conflicts(
        self, 
        class_id: int, 
        start_slot: int, 
        room_id: int, 
        chromosome: Dict[int, Dict[str, int]],
        room_to_classes: Dict[int, List[int]] = None
    ) -> float:
        """
        Fast evaluation of the penalty score for assigning class_id to start_slot and room_id,
        assuming the rest of the chromosome remains unchanged.
        """
        cl = self.ds.classes[class_id]
        cl_len = self.ds.class_lengths[class_id]
        cl_end = start_slot + cl_len
        
        student_errors = 0
        room_overlap_errors = 0
        capacity_errors = 0
        time_errors = 0
        
        # 1. Student overlaps
        partners = self.class_conflict_partners.get(class_id, [])
        for other_cid, shared in partners:
            if other_cid == class_id:
                continue
            other_gene = chromosome[other_cid]
            other_slot = other_gene["start_slot"]
            other_end = other_slot + self.ds.class_lengths[other_cid]
            if start_slot < other_end and other_slot < cl_end:
                student_errors += shared
                
        # 2. Room overlap (optimized using room_to_classes map)
        if room_to_classes is not None:
            for other_cid in room_to_classes.get(room_id, []):
                if other_cid == class_id:
                    continue
                other_gene = chromosome[other_cid]
                other_slot = other_gene["start_slot"]
                other_end = other_slot + self.ds.class_lengths[other_cid]
                if start_slot < other_end and other_slot < cl_end:
                    room_overlap_errors += 1
        else:
            for other_cid, other_gene in chromosome.items():
                if other_cid == class_id:
                    continue
                if other_gene["room_id"] == room_id:
                    other_slot = other_gene["start_slot"]
                    other_end = other_slot + self.ds.class_lengths[other_cid]
                    if start_slot < other_end and other_slot < cl_end:
                        room_overlap_errors += 1
                    
        # 3. Capacity shortage
        capacity_diff = self.ds.class_enrollments[class_id] - self.ds.room_capacities[room_id]
        if capacity_diff > 0:
            capacity_errors = capacity_diff
            
        # 4. Allowed time slot check
        allowed_set = self.ds.allowed_slots[class_id]
        if allowed_set is not None and start_slot not in allowed_set:
            time_errors = 1
            
        student_weight = self.ds.weights.get("student", 5.0)
        room_weight = self.ds.weights.get("room", 1.0)
        time_weight = self.ds.weights.get("time", 4.0)
        
        local_penalty = (
            (student_errors * student_weight) +
            ((room_overlap_errors + capacity_errors) * room_weight) +
            (time_errors * time_weight)
        )
        return float(local_penalty)

    def local_search_mutate(self, chromosome: Dict[int, Dict[str, int]]) -> Dict[int, Dict[str, int]]:
        mutated = {cid: gene.copy() for cid, gene in chromosome.items()}
        
        # Build room to classes index
        room_to_classes = {}
        for cid, gene in mutated.items():
            room_to_classes.setdefault(gene["room_id"], []).append(cid)
            
        # Identify violating/high-penalty classes in this chromosome.
        violating_classes = []
        for class_id in self.ds.class_ids:
            gene = mutated[class_id]
            slot = gene["start_slot"]
            room = gene["room_id"]
            
            # Simple quick check: check time constraint
            allowed_set = self.ds.allowed_slots[class_id]
            if allowed_set is not None and slot not in allowed_set:
                violating_classes.append(class_id)
                continue
                
            # Check room overlap (fast: only search classes assigned to the same room)
            cl_len = self.ds.class_lengths[class_id]
            cl_end = slot + cl_len
            for other_cid in room_to_classes.get(room, []):
                if other_cid == class_id:
                    continue
                other_gene = mutated[other_cid]
                other_slot = other_gene["start_slot"]
                other_end = other_slot + self.ds.class_lengths[other_cid]
                if slot < other_end and other_slot < cl_end:
                    violating_classes.append(class_id)
                    break
                        
        # If no classes are currently violating, choose a random one
        if not violating_classes:
            class_to_mutate = random.choice(self.ds.class_ids)
        else:
            class_to_mutate = random.choice(violating_classes)
            
        cl = self.ds.classes[class_to_mutate]
        max_start = max(0, self.ds.total_slots - cl.length)
        allowed_rooms = cl.allowed_rooms if cl.allowed_rooms else self.ds.room_ids
        enrollment = self.ds.class_enrollments[class_to_mutate]
        fit_rooms = [r_id for r_id in allowed_rooms if self.ds.room_capacities[r_id] >= enrollment]
        rooms_to_try = fit_rooms if fit_rooms else allowed_rooms
        
        # Try 15 candidates and pick the one with the lowest local conflicts
        best_slot = mutated[class_to_mutate]["start_slot"]
        best_room = mutated[class_to_mutate]["room_id"]
        best_score = self._evaluate_local_conflicts(class_to_mutate, best_slot, best_room, mutated, room_to_classes)
        
        candidates = []
        if cl.allowed_times:
            allowed_slots_list = list(self.ds.allowed_slots[class_to_mutate]) if self.ds.allowed_slots[class_to_mutate] else []
            if allowed_slots_list:
                for _ in range(min(15, len(allowed_slots_list))):
                    slot = random.choice(allowed_slots_list)
                    room = random.choice(rooms_to_try)
                    candidates.append(("move", slot, room, None))
                    
        if not candidates:
            for _ in range(15):
                slot = random.randint(0, max_start)
                room = random.choice(rooms_to_try)
                candidates.append(("move", slot, room, None))
                
        # Add 5 swap candidates
        other_classes = random.sample(self.ds.class_ids, min(5, len(self.ds.class_ids)))
        for other_cid in other_classes:
            if other_cid == class_to_mutate:
                continue
            other_gene = mutated[other_cid]
            other_cl = self.ds.classes[other_cid]
            if other_gene["start_slot"] <= max_start and mutated[class_to_mutate]["start_slot"] <= max(0, self.ds.total_slots - other_cl.length):
                candidates.append(("swap", other_gene["start_slot"], other_gene["room_id"], other_cid))
                
        for c_type, slot, room, other_cid in candidates:
            if c_type == "move":
                score = self._evaluate_local_conflicts(class_to_mutate, slot, room, mutated, room_to_classes)
                if score < best_score:
                    best_score = score
                    # Temp revert index
                    old_room = mutated[class_to_mutate]["room_id"]
                    if old_room in room_to_classes and class_to_mutate in room_to_classes[old_room]:
                        room_to_classes[old_room].remove(class_to_mutate)
                    mutated[class_to_mutate]["start_slot"] = slot
                    mutated[class_to_mutate]["room_id"] = room
                    room_to_classes.setdefault(room, []).append(class_to_mutate)
                    best_slot = slot
                    best_room = room
                    
            elif c_type == "swap":
                old_c1_slot = mutated[class_to_mutate]["start_slot"]
                old_c1_room = mutated[class_to_mutate]["room_id"]
                
                old_c2_slot = mutated[other_cid]["start_slot"]
                old_c2_room = mutated[other_cid]["room_id"]
                
                score_before = (
                    self._evaluate_local_conflicts(class_to_mutate, old_c1_slot, old_c1_room, mutated, room_to_classes) +
                    self._evaluate_local_conflicts(other_cid, old_c2_slot, old_c2_room, mutated, room_to_classes)
                )
                
                # Temp swap in mutated and index
                mutated[class_to_mutate]["start_slot"] = slot
                mutated[class_to_mutate]["room_id"] = room
                mutated[other_cid]["start_slot"] = old_c1_slot
                mutated[other_cid]["room_id"] = old_c1_room
                
                if old_c1_room in room_to_classes and class_to_mutate in room_to_classes[old_c1_room]:
                    room_to_classes[old_c1_room].remove(class_to_mutate)
                if old_c2_room in room_to_classes and other_cid in room_to_classes[old_c2_room]:
                    room_to_classes[old_c2_room].remove(other_cid)
                room_to_classes.setdefault(room, []).append(class_to_mutate)
                room_to_classes.setdefault(old_c1_room, []).append(other_cid)
                
                score_after = (
                    self._evaluate_local_conflicts(class_to_mutate, slot, room, mutated, room_to_classes) +
                    self._evaluate_local_conflicts(other_cid, old_c1_slot, old_c1_room, mutated, room_to_classes)
                )
                
                if score_after < score_before:
                    # Keep the swap!
                    best_slot = slot
                    best_room = room
                    best_score = self._evaluate_local_conflicts(class_to_mutate, best_slot, best_room, mutated, room_to_classes)
                else:
                    # Revert swap and index
                    mutated[class_to_mutate]["start_slot"] = old_c1_slot
                    mutated[class_to_mutate]["room_id"] = old_c1_room
                    mutated[other_cid]["start_slot"] = old_c2_slot
                    mutated[other_cid]["room_id"] = old_c2_room
                    
                    if class_to_mutate in room_to_classes.get(room, []):
                        room_to_classes[room].remove(class_to_mutate)
                    if other_cid in room_to_classes.get(old_c1_room, []):
                        room_to_classes[old_c1_room].remove(other_cid)
                    room_to_classes.setdefault(old_c1_room, []).append(class_to_mutate)
                    room_to_classes.setdefault(old_c2_room, []).append(other_cid)
                    
        return mutated

    def swap_mutate(self, chromosome: Dict[int, Dict[str, int]]) -> Dict[int, Dict[str, int]]:
        mutated = {cid: gene.copy() for cid, gene in chromosome.items()}
        if len(self.ds.class_ids) < 2:
            return mutated
        c1, c2 = random.sample(self.ds.class_ids, 2)
        
        gene1 = mutated[c1]
        gene2 = mutated[c2]
        
        cl1 = self.ds.classes[c1]
        cl2 = self.ds.classes[c2]
        
        slot1_new = min(gene2["start_slot"], max(0, self.ds.total_slots - cl1.length))
        slot2_new = min(gene1["start_slot"], max(0, self.ds.total_slots - cl2.length))
        
        room1_new = gene2["room_id"]
        room2_new = gene1["room_id"]
        
        mutated[c1] = {
            "start_slot": slot1_new,
            "room_id": room1_new
        }
        mutated[c2] = {
            "start_slot": slot2_new,
            "room_id": room2_new
        }
        return mutated

    def mutate(self, chromosome: Dict[int, Dict[str, int]]) -> Dict[int, Dict[str, int]]:
        # Choose mutation strategy:
        # 30% Guided local search mutation, 30% Swap mutation, 40% standard random mutation
        r = random.random()
        if r < 0.30:
            return self.local_search_mutate(chromosome)
        elif r < 0.60:
            return self.swap_mutate(chromosome)
            
        # Otherwise, fall back to standard random mutation
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
                allowed_rooms = cl.allowed_rooms if cl.allowed_rooms else self.ds.room_ids
                enrollment = self.ds.class_enrollments[class_id]
                fit_rooms = [r_id for r_id in allowed_rooms if self.ds.room_capacities[r_id] >= enrollment]
                if fit_rooms:
                    mutated[class_id]["room_id"] = random.choice(fit_rooms)
                else:
                    mutated[class_id]["room_id"] = max(allowed_rooms, key=lambda r_id: self.ds.room_capacities[r_id])
                    
        return mutated

    def _evaluate(self, chromosome: Dict[int, Dict[str, int]]) -> EvaluatedChromosome:
        fitness, scores = calculate_fitness(chromosome, self.ds)
        return fitness, scores["total_penalty"], scores, chromosome

    def _evaluate_population(self) -> List[EvaluatedChromosome]:
        evaluated = [self._evaluate(chrom) for chrom in self.population]
        # Sort by total penalty (ascending, lower is better)
        evaluated.sort(key=lambda item: item[1])
        return evaluated

    def _tournament_selection(self, evaluated_population: List[EvaluatedChromosome], t_size: int = None) -> Dict[int, Dict[str, int]]:
        if t_size is None:
            t_size = getattr(self, "current_tournament_size", self.tournament_size)
        size = min(t_size, len(evaluated_population))
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

    def repair_chromosome(self, chromosome: Dict[int, Dict[str, int]], generation: int = 1, total_generations: int = 100) -> Dict[int, Dict[str, int]]:
        """
        Runs a local search with Simulated Annealing acceptance to repair room/student overlaps 
        and capacity violations on the best individual.
        """
        import math
        repaired = {cid: gene.copy() for cid, gene in chromosome.items()}
        
        # Build room to classes index
        room_to_classes = {}
        for cid, gene in repaired.items():
            room_to_classes.setdefault(gene["room_id"], []).append(cid)
            
        # Find violating classes (with room overlaps or capacity violations)
        violating_classes = []
        for class_id in self.ds.class_ids:
            gene = repaired[class_id]
            slot = gene["start_slot"]
            room = gene["room_id"]
            
            # Check capacity
            if self.ds.class_enrollments[class_id] > self.ds.room_capacities[room]:
                violating_classes.append(class_id)
                continue
                
            # Check time allowed slot
            allowed_set = self.ds.allowed_slots[class_id]
            if allowed_set is not None and slot not in allowed_set:
                violating_classes.append(class_id)
                continue
                
            # Check room overlap
            cl_len = self.ds.class_lengths[class_id]
            cl_end = slot + cl_len
            for other_cid in room_to_classes.get(room, []):
                if other_cid == class_id:
                    continue
                other_gene = repaired[other_cid]
                other_slot = other_gene["start_slot"]
                other_end = other_slot + self.ds.class_lengths[other_cid]
                if slot < other_end and other_slot < cl_end:
                    violating_classes.append(class_id)
                    break
                    
        # Shuffle violating classes to run in random order
        random.shuffle(violating_classes)
        
        # Simulated Annealing setup: cooling down over generations
        progress = generation / max(1, total_generations)
        T = 50.0 * (1.0 - progress)
        T = max(1e-6, T)
        
        # Limit local search to at most 100 violating classes per generation to keep it fast
        for class_id in violating_classes[:100]:
            cl = self.ds.classes[class_id]
            max_start = max(0, self.ds.total_slots - cl.length)
            
            best_slot = repaired[class_id]["start_slot"]
            best_room = repaired[class_id]["room_id"]
            best_score = self._evaluate_local_conflicts(class_id, best_slot, best_room, repaired, room_to_classes)
            
            # If it already has 0 local penalty, skip
            if best_score == 0:
                continue
                
            # Determine candidates: filter rooms by capacity
            allowed_rooms = cl.allowed_rooms if cl.allowed_rooms else self.ds.room_ids
            class_enrollment = self.ds.class_enrollments[class_id]
            fit_rooms = [r_id for r_id in allowed_rooms if self.ds.room_capacities[r_id] >= class_enrollment]
            rooms_to_try = fit_rooms if fit_rooms else allowed_rooms
            
            candidates = []
            # 1. Single-class move candidates
            if cl.allowed_times:
                allowed_slots_list = list(self.ds.allowed_slots[class_id]) if self.ds.allowed_slots[class_id] else []
                if allowed_slots_list:
                    for _ in range(min(5, len(allowed_slots_list))):
                        slot = random.choice(allowed_slots_list)
                        room = random.choice(rooms_to_try)
                        candidates.append(("move", slot, room, None))
            else:
                for _ in range(5):
                    slot = random.randint(0, max_start)
                    room = random.choice(rooms_to_try)
                    candidates.append(("move", slot, room, None))
                    
            # 2. Swap candidates: select 5 random other classes to swap with
            other_classes = random.sample(self.ds.class_ids, min(5, len(self.ds.class_ids)))
            for other_cid in other_classes:
                if other_cid == class_id:
                    continue
                other_gene = repaired[other_cid]
                other_cl = self.ds.classes[other_cid]
                if other_gene["start_slot"] <= max_start and repaired[class_id]["start_slot"] <= max(0, self.ds.total_slots - other_cl.length):
                    candidates.append(("swap", other_gene["start_slot"], other_gene["room_id"], other_cid))
                    
            for c_type, slot, room, other_cid in candidates:
                if c_type == "move":
                    score = self._evaluate_local_conflicts(class_id, slot, room, repaired, room_to_classes)
                    delta = score - best_score
                    
                    if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / T)):
                        # Accept move (better immediately, or worse with SA probability)
                        old_room = repaired[class_id]["room_id"]
                        if old_room in room_to_classes and class_id in room_to_classes[old_room]:
                            room_to_classes[old_room].remove(class_id)
                        repaired[class_id]["start_slot"] = slot
                        repaired[class_id]["room_id"] = room
                        room_to_classes.setdefault(room, []).append(class_id)
                        best_slot = slot
                        best_room = room
                        best_score = score
                        
                elif c_type == "swap":
                    old_c1_slot = repaired[class_id]["start_slot"]
                    old_c1_room = repaired[class_id]["room_id"]
                    
                    old_c2_slot = repaired[other_cid]["start_slot"]
                    old_c2_room = repaired[other_cid]["room_id"]
                    
                    score_before = (
                        self._evaluate_local_conflicts(class_id, old_c1_slot, old_c1_room, repaired, room_to_classes) +
                        self._evaluate_local_conflicts(other_cid, old_c2_slot, old_c2_room, repaired, room_to_classes)
                    )
                    
                    # Temp swap
                    repaired[class_id]["start_slot"] = slot
                    repaired[class_id]["room_id"] = room
                    repaired[other_cid]["start_slot"] = old_c1_slot
                    repaired[other_cid]["room_id"] = old_c1_room
                    
                    # Temp update room_to_classes index
                    if old_c1_room in room_to_classes and class_id in room_to_classes[old_c1_room]:
                        room_to_classes[old_c1_room].remove(class_id)
                    if old_c2_room in room_to_classes and other_cid in room_to_classes[old_c2_room]:
                        room_to_classes[old_c2_room].remove(other_cid)
                    room_to_classes.setdefault(room, []).append(class_id)
                    room_to_classes.setdefault(old_c1_room, []).append(other_cid)
                    
                    score_after = (
                        self._evaluate_local_conflicts(class_id, slot, room, repaired, room_to_classes) +
                        self._evaluate_local_conflicts(other_cid, old_c1_slot, old_c1_room, repaired, room_to_classes)
                    )
                    
                    delta = score_after - score_before
                    
                    if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / T)):
                        # Accept the swap!
                        best_slot = slot
                        best_room = room
                        best_score = self._evaluate_local_conflicts(class_id, best_slot, best_room, repaired, room_to_classes)
                    else:
                        # Revert swap in repaired
                        repaired[class_id]["start_slot"] = old_c1_slot
                        repaired[class_id]["room_id"] = old_c1_room
                        repaired[other_cid]["start_slot"] = old_c2_slot
                        repaired[other_cid]["room_id"] = old_c2_room
                        
                        # Revert index
                        if class_id in room_to_classes.get(room, []):
                            room_to_classes[room].remove(class_id)
                        if other_cid in room_to_classes.get(old_c1_room, []):
                            room_to_classes[old_c1_room].remove(other_cid)
                        room_to_classes.setdefault(old_c1_room, []).append(class_id)
                        room_to_classes.setdefault(old_c2_room, []).append(other_cid)
                        
        return repaired

    def run(self, generations: int = 100, resume: bool = False) -> Dict[int, Dict[str, int]]:
        if not resume:
            self.history = []
            if not self.population:
                self.initialize_population()
            evaluated = self._evaluate_population()
            best_overall = evaluated[0]
            self._log_generation(0, best_overall)
        else:
            evaluated = self._evaluate_population()
            best_overall = evaluated[0]
            
        elite_count = max(1, int(self.population_size * self.elitism_rate))
        
        # Adaptive parameters setup
        base_mutation_rate = self.mutation_rate
        stagnant_count = 0
        
        start_gen = len(self.history) if resume else 1
        
        for generation in range(1, generations + 1):
            current_gen_num = start_gen + generation if resume else generation
            total_expected_gens = start_gen + generations if resume else generations
            
            # 1. Adaptive Tournament Size (selection pressure increases over time)
            # Starts at 5, linearly scales up to self.tournament_size by 50% of the generations
            t_size = int(5 + (self.tournament_size - 5) * min(1.0, current_gen_num / max(1, total_expected_gens / 2)))
            self.current_tournament_size = max(2, min(self.tournament_size, t_size))
            
            # Elitism: carry over the best individuals directly
            next_population = [
                {cid: gene.copy() for cid, gene in item[3].items()}
                for item in evaluated[:elite_count]
            ]
            
            # Selection, Crossover, and Mutation
            while len(next_population) < self.population_size:
                parent1 = self._tournament_selection(evaluated, self.current_tournament_size)
                parent2 = self._tournament_selection(evaluated, self.current_tournament_size)
                child1, child2 = self.crossover(parent1, parent2)
                
                next_population.append(self.mutate(child1))
                if len(next_population) < self.population_size:
                    next_population.append(self.mutate(child2))
                    
            self.population = next_population
            evaluated = self._evaluate_population()
            
            # Perform local repair on the best chromosome in the population
            best_chrom_repaired = self.repair_chromosome(evaluated[0][3], generation=current_gen_num, total_generations=total_expected_gens)
            repaired_fitness, repaired_penalty, repaired_scores, _ = self._evaluate(best_chrom_repaired)
            
            # If the repair improved the fitness, inject it back as the first element of the population
            if repaired_penalty < evaluated[0][1]:
                evaluated[0] = (repaired_fitness, repaired_penalty, repaired_scores, best_chrom_repaired)
                best_idx = 0
                for idx, chrom in enumerate(self.population):
                    if chrom is evaluated[0][3]:
                        best_idx = idx
                        break
                self.population[best_idx] = best_chrom_repaired
            
            if evaluated[0][1] < best_overall[1]:
                best_overall = evaluated[0]
                stagnant_count = 0
                self.mutation_rate = base_mutation_rate # Reset mutation rate
            else:
                stagnant_count += 1
                
            # Adaptive Mutation: if stuck, boost mutation rate by 50%, else decay back to base
            if stagnant_count >= 5:
                self.mutation_rate = min(0.50, base_mutation_rate * 1.5)
            else:
                self.mutation_rate = max(base_mutation_rate, self.mutation_rate * 0.95)
                
            self._log_generation(current_gen_num, evaluated[0])
            
            # Early stopping if optimal zero-penalty schedule found
            if best_overall[1] == 0:
                logger.info("Stopping early: zero-penalty chromosome found.")
                break
                
        # Restore original mutation rate when finished
        self.mutation_rate = base_mutation_rate
        return {cid: gene.copy() for cid, gene in best_overall[3].items()}

    def run_with_scores(self, generations: int = 100) -> EvaluatedChromosome:
        best_chromosome = self.run(generations)
        return self._evaluate(best_chromosome)
