from typing import List, Tuple, Dict
from .model import Dataset
import numpy as np
import math

class FitnessCalculator:
    def __init__(self, dataset: Dataset):
        self.ds = dataset
        self.weights = {
            "direct_conflict": 1000.0,
            "large_exams": 2500000.0,
            "more_than_2_a_day": 100.0,
            "back_to_back": 10.0,
            "room_split": 10.0,
            "period_penalty": 1.0,
            "room_penalty": 1.0,
            "room_split_distance": 0.01,
            "room_size": 0.001,
            "room_distance": 0.0001,
            "rotation_penalty": 0.0001
        }
        
        # Pre-compute room distances
        self._distance_cache = {}
        room_ids = list(self.ds.rooms.keys())
        for r1 in room_ids:
            self._distance_cache[r1] = {}
            for r2 in room_ids:
                if r1 == r2:
                    self._distance_cache[r1][r2] = 0.0
                else:
                    self._distance_cache[r1][r2] = self._haversine(
                        self.ds.rooms[r1].lat, self.ds.rooms[r1].lon,
                        self.ds.rooms[r2].lat, self.ds.rooms[r2].lon
                    )
                    
        # Pre-compute back-to-back periods
        self._b2b_periods = []
        for p in range(1, 29):
            if self.ds.periods[p].day == self.ds.periods[p+1].day:
                self._b2b_periods.append((p, p+1))
                
        # Pre-compute students taking 3+ exams
        self._students_3_plus = [exams for exams in self.ds.student_exams.values() if len(exams) >= 3]
        
        # Map period id to a day index (0 to 5 for Mon to Sat)
        self._period_day_idx = [0]*30
        days = []
        for p in range(1, 30):
            d = self.ds.periods[p].day
            if d not in days:
                days.append(d)
            self._period_day_idx[p] = days.index(d)
        
    def _haversine(self, lat1, lon1, lat2, lon2):
        # Calculate distance in meters
        R = 6371000 # radius of earth in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi / 2.0) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * \
            math.sin(delta_lambda / 2.0) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def calculate_fitness(self, chromosome: List[Tuple[int, Tuple[int, ...]]]) -> Tuple[float, Dict[str, float]]:
        scores = {k: 0.0 for k in self.weights}
        
        # 1. Large Exams Spilling
        for idx in self.ds.large_exams_indices:
            if chromosome[idx][0] > 24:
                scores["large_exams"] += self.weights["large_exams"]
                
        # 2. Period Penalty & Room Penalty & Room Split & Room Size & Room Split Distance
        for idx, exam in enumerate(self.ds.exams):
            period_id, room_ids = chromosome[idx]
            
            # Period Penalty
            p_penalty = self.ds.periods[period_id].penalty
            scores["period_penalty"] += p_penalty * self.weights["period_penalty"]
            
            # Room Split Penalty
            num_rooms = len(room_ids)
            if num_rooms > 1:
                scores["room_split"] += self.weights["room_split"]
                # Room Split Distance
                max_dist = 0.0
                for i in range(num_rooms):
                    r1 = room_ids[i]
                    for j in range(i + 1, num_rooms):
                        dist = self._distance_cache[r1][room_ids[j]]
                        if dist > max_dist:
                            max_dist = dist
                scores["room_split_distance"] += max_dist * self.weights["room_split_distance"]
            
            # Room Penalty & Size Penalty
            total_capacity = 0
            for rid in room_ids:
                room = self.ds.rooms[rid]
                r_penalty = room.penalty_periods.get(period_id, 0)
                scores["room_penalty"] += r_penalty * self.weights["room_penalty"]
                cap = room.alt_size if exam.alt_seating else room.size
                total_capacity += cap
                
            # Room Size inefficient use
            if total_capacity > exam.students_count:
                scores["room_size"] += (total_capacity - exam.students_count) * self.weights["room_size"]
        
        # 3. Matrix based student conflicts
        period_exams = [[] for _ in range(30)]
        for idx, (p_id, _) in enumerate(chromosome):
            period_exams[p_id].append(idx)
            
        # Direct conflict
        for p_id in range(1, 30):
            exams = period_exams[p_id]
            n = len(exams)
            for i in range(n):
                ex1 = exams[i]
                for j in range(i + 1, n):
                    conflicts = self.ds.conflict_matrix[ex1][exams[j]]
                    if conflicts > 0:
                        scores["direct_conflict"] += conflicts * self.weights["direct_conflict"]
                        
        # Back to back & room distance
        for p1, p2 in self._b2b_periods:
            for ex1 in period_exams[p1]:
                for ex2 in period_exams[p2]:
                    conflicts = self.ds.conflict_matrix[ex1][ex2]
                    if conflicts > 0:
                        scores["back_to_back"] += conflicts * self.weights["back_to_back"]
                        
                        max_d = 0.0
                        for r1 in chromosome[ex1][1]:
                            for r2 in chromosome[ex2][1]:
                                d = self._distance_cache[r1][r2]
                                if d > max_d: max_d = d
                        scores["room_distance"] += conflicts * max_d * self.weights["room_distance"]
                        
        # More than 2 a day
        for exams in self._students_3_plus:
            day_counts = [0] * 6
            for idx in exams:
                day_counts[self._period_day_idx[chromosome[idx][0]]] += 1
                
            for count in day_counts:
                if count > 2:
                    scores["more_than_2_a_day"] += self.weights["more_than_2_a_day"]

        total_penalty = sum(scores.values())
        return total_penalty, scores
