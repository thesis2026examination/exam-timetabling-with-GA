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
        
        # Güvenli taban oranı koruyoruz. Eğer dışarıdan çok yüksek (örneğin 0.1) 
        # verilirse otomatik olarak makul bir seviyeye (0.015) sabitliyoruz.
        self.base_mutation_rate = mutation_rate if mutation_rate < 0.05 else 0.015
        self.mutation_rate = self.base_mutation_rate  
        
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
                best_periods, min_conf = [], float('inf')
                for p in valid_periods:
                    others = period_exams[p]
                    conflicts = int(self.ds.conflict_matrix[exam.idx, others].sum()) if others else 0
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
        Conflict-Aware Uniform Crossover.
        Ebeveynlerin gen haritalarını rastgele bölmek yerine, çakışması (conflict) 
        daha az olan yerleşimleri öncelikli olarak çocuklara aktarır.
        """
        child1 = [None] * len(parent1)
        child2 = [None] * len(parent2)
        
        # Hızlı conflict kontrolü için ebeveynlerin periyot şemalarını çıkar
        p1_periods = [[] for _ in range(30)]
        p2_periods = [[] for _ in range(30)]
        for idx in range(len(parent1)):
            p1_periods[parent1[idx][0]].append(idx)
            p2_periods[parent2[idx][0]].append(idx)

        for idx in range(len(parent1)):
            # Eğer gen kısıtlıysa doğrudan geçir
            if idx in self.ds.time_fixed_exams:
                child1[idx] = parent1[idx]
                child2[idx] = parent2[idx]
                continue

            # NumPy ile iki ebeveyndeki lokal çakışma maliyetlerini hesapla
            p1_pos, p2_pos = parent1[idx][0], parent2[idx][0]
            
            c1_others = [o for o in p1_periods[p1_pos] if o != idx]
            c2_others = [o for o in p2_periods[p2_pos] if o != idx]
            
            conf1 = int(self.ds.conflict_matrix[idx, c1_others].sum()) if c1_others else 0
            conf2 = int(self.ds.conflict_matrix[idx, c2_others].sum()) if c2_others else 0

            # Baskın (daha az çakışmalı) olan genleri çocuklara akıllıca dağıt
            if conf1 < conf2:
                child1[idx] = parent1[idx]
                child2[idx] = parent1[idx] if random.random() < 0.3 else parent2[idx]
            elif conf2 < conf1:
                child1[idx] = parent2[idx]
                child2[idx] = parent2[idx] if random.random() < 0.3 else parent1[idx]
            else:
                # Eşitlik durumunda uniform dağılım
                if random.random() < 0.5:
                    child1[idx] = parent1[idx]
                    child2[idx] = parent2[idx]
                else:
                    child1[idx] = parent2[idx]
                    child2[idx] = parent1[idx]
                
        return child1, child2
        
    def mutate(self, chromosome: List[Tuple[int, Tuple[int, ...]]]) -> List[Tuple[int, Tuple[int, ...]]]:
        """
        Multi-Strategy Mutation Framework.
        Sadece period değiştirmek yerine, Swap ve Room mutasyonları ile popülasyonu tıkanmaktan kurtarır.
        """
        num_mutations = np.random.binomial(len(chromosome), self.mutation_rate)
        if num_mutations == 0:
            return chromosome

        new_chromosome = chromosome.copy()
        mutation_indices = random.sample(range(len(chromosome)), num_mutations)

        # Güncel period->exam haritası
        period_exams = [[] for _ in range(30)]
        for i, (p_id, _) in enumerate(new_chromosome):
            period_exams[p_id].append(i)

        for idx in mutation_indices:
            is_time_fixed = idx in self.ds.time_fixed_exams
            is_room_fixed = idx in self.ds.room_fixed_exams

            if is_time_fixed and is_room_fixed:
                continue

            # %30 ihtimalle SWAP Mutasyonu uygula (İki sınavın yerini değiştir - Yapıyı korur)
            if not is_time_fixed and random.random() < 0.30:
                swap_with = random.randint(0, len(new_chromosome) - 1)
                if swap_with != idx and swap_with not in self.ds.time_fixed_exams:
                    p_id_1, rooms_1 = new_chromosome[idx]
                    p_id_2, rooms_2 = new_chromosome[swap_with]
                    
                    # Swap işlemi
                    new_chromosome[idx] = (p_id_2, rooms_1)
                    new_chromosome[swap_with] = (p_id_1, rooms_2)
                    
                    # Haritayı güncelle
                    period_exams[p_id_1].remove(idx)
                    period_exams[p_id_2].append(idx)
                    if swap_with in period_exams[p_id_2]:
                        period_exams[p_id_2].remove(swap_with)
                    period_exams[p_id_1].append(swap_with)
                    continue

            p_id, rooms = new_chromosome[idx]

            # 1. PERIOD MUTATION — Conflict-Aware Greedy Selection
            if not is_time_fixed:
                exam = self.ds.exams[idx]
                if idx in self.large_exam_ids:
                    opts = [p for p in range(1, 24) if not exam.available_periods or p in exam.available_periods]
                else:
                    opts = list(exam.available_periods) if exam.available_periods else list(self.ds.periods.keys())

                if opts:
                    if idx in period_exams[p_id]:
                        period_exams[p_id].remove(idx)

                    best_p, min_conf = p_id, float('inf')
                    for p in opts:
                        others = period_exams[p]
                        conf = int(self.ds.conflict_matrix[idx, others].sum()) if others else 0
                        if conf < min_conf:
                            min_conf = conf
                            best_p = p
                            if conf == 0:
                                break

                    p_id = best_p
                    period_exams[p_id].append(idx)

            # 2. ROOM MUTATION — Smart Capacity Allocation
            if not is_room_fixed:
                exam = self.ds.exams[idx]
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

                rooms = tuple(assigned_rooms)

            new_chromosome[idx] = (p_id, rooms)

        return new_chromosome

    def _tournament_selection(self, evaluated_population: List[Tuple[float, List[Tuple[int, Tuple[int, ...]]]]], tournament_size: int = 7) -> List[Tuple[int, Tuple[int, ...]]]:
        """Tournament size increased to 7 to enforce higher selection pressure towards better schemas."""
        tournament = random.sample(evaluated_population, tournament_size)
        tournament.sort(key=lambda x: x[0])  
        return tournament[0][1]
        
    def _repair_operator(self, chromosome: List[Tuple[int, Tuple[int, ...]]]) -> List[Tuple[int, Tuple[int, ...]]]:
        """
        Detects top bottleneck exams with intense direct conflicts using vectorized NumPy slices
        and hot-swaps them into the most peaceful valid periods.
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
            return repaired

        # Evacuate top 60 worst offending exams to minimize penalty mass rapidly
        sorted_exams = sorted(exam_conflicts.items(), key=lambda x: x[1], reverse=True)
        top_exams = [ex for ex, _ in sorted_exams[:60]]

        for ex in top_exams:
            if ex in self.ds.time_fixed_exams:
                continue

            valid_periods = (list(self.ds.exams[ex].available_periods)
                             if self.ds.exams[ex].available_periods
                             else list(self.ds.periods.keys()))
            if ex in self.large_exam_ids:
                valid_periods = [p for p in valid_periods if p < 24]
            if not valid_periods:
                continue

            best_p = repaired[ex][0]
            min_conflicts = exam_conflicts.get(ex, float('inf'))

            old_p = repaired[ex][0]
            if ex in period_exams[old_p]:
                period_exams[old_p].remove(ex)

            for p_id in valid_periods:
                others = period_exams[p_id]
                c = int(self.ds.conflict_matrix[ex, others].sum()) if others else 0
                if c < min_conflicts:
                    min_conflicts = c
                    best_p = p_id
                    if c == 0:
                        break  

            repaired[ex] = (best_p, repaired[ex][1])
            period_exams[best_p].append(ex)

        return repaired

    def run(self, generations: int = 100):
        self.initialize_population()

        # --- Random Shock Hyper-Mutation State ---
        patience         = 4               # Tıkanma kontrolü periyodu
        stagnation       = 0
        last_best        = float('inf')
        # ---

        for gen in range(generations):
            evaluated = []
            for chrom in self.population:
                fitness, _ = self.fitness_calc.calculate_fitness(chrom)
                evaluated.append((fitness, chrom))

            evaluated.sort(key=lambda x: x[0])

            # Repair Operator: Top %12 population to boost building block generation rate
            repair_count = max(1, int(self.population_size * 0.12))
            for i in range(repair_count):
                repaired = self._repair_operator(evaluated[i][1])
                rep_fitness, _ = self.fitness_calc.calculate_fitness(repaired)
                if rep_fitness < evaluated[i][0]:
                    evaluated[i] = (rep_fitness, repaired)

            evaluated.sort(key=lambda x: x[0])
            best_fitness = evaluated[0][0]

            # --- Dinamik Kaçış Stratejisi: Adaptif Mutasyon + Cataclysm (Kitlesel Yok Oluş) ---
            if best_fitness < last_best:
                stagnation = 0
                self.mutation_rate = self.base_mutation_rate
            else:
                stagnation += 1

            cataclysm_triggered = False
            
            if stagnation >= patience * 4: # Örneğin 16 jenerasyon (patience=4)
                # KİTLESEL YOK OLUŞ (CATACLYSM)
                # Çok uzun süre tıkanırsa mutasyon oranı işe yaramaz. Popülasyona taze kan gerekir.
                logger.warning(f"Generation {gen}: [CATACLYSM] {stagnation} jenerasyondur ilerleme yok! Popülasyonun büyük kısmı yenileniyor.")
                self.mutation_rate = self.base_mutation_rate
                stagnation = 0
                cataclysm_triggered = True
            elif stagnation >= patience:
                # Kademeli artış ama YIKICI OLMAYACAK seviyede (max ~0.08 arası)
                # 0.50 gibi oranlar çocukları tamamen çöpe çevirir ve sadece elitlerin hayatta kalmasını sağlar.
                self.mutation_rate = min(0.08, self.base_mutation_rate + (stagnation - patience) * 0.005)
                logger.info(f"Generation {gen}: [ADAPTIVE] Stagnation {stagnation}. mut_rate hafif artırıldı: {self.mutation_rate:.4f}")
            else:
                self.mutation_rate = self.base_mutation_rate

            last_best = best_fitness

            logger.info(
                f"Generation {gen}: Best Fitness = {best_fitness:.2f} | "
                f"mut_rate={self.mutation_rate:.4f} | stagnation={stagnation}/{patience}"
            )

            # Elitism: Normalde %10 koru. Cataclysm anında sadece EN İYİ 1 bireyi koru!
            if cataclysm_triggered:
                elitism_count = 1
            else:
                elitism_count = max(4, int(self.population_size * 0.10))
                
            next_population = [chrom for _, chrom in evaluated[:elitism_count]]

            # Cataclysm tetiklendiyse, popülasyonun %60'ını tamamen sıfırdan, akıllı sezgisel fonksiyonla üret
            if cataclysm_triggered:
                fresh_count = int(self.population_size * 0.60)
                for _ in range(fresh_count):
                    next_population.append(self.generate_random_chromosome())

            # Reproduction Loop
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

        # Final Evaluation
        evaluated = []
        for chrom in self.population:
            fitness, _ = self.fitness_calc.calculate_fitness(chrom)
            evaluated.append((fitness, chrom))
        evaluated.sort(key=lambda x: x[0])

        return evaluated[0][1]