import logging
import numpy as np
from typing import Dict, Tuple
from .dataset_parser import XMLDataset

logger = logging.getLogger(__name__)

# Chromosome structure: Dict[int, Dict[str, int]]
# e.g., { class_id: {"start_slot": int, "room_id": int} }

def calculate_fitness(chromosome: Dict[int, Dict[str, int]], dataset: XMLDataset) -> Tuple[float, Dict[str, int]]:
    """
    Highly-optimized fitness calculation with NumPy vectorization.
    """
    student_errors = 0
    room_overlap_errors = 0
    capacity_errors = 0
    travel_errors = 0
    time_errors = 0
    dist_errors = 0
    
    # 1. Populate starts, ends, and rooms using flat precomputations
    starts = np.zeros(dataset.max_class_id + 1, dtype=np.int32)
    ends = np.zeros(dataset.max_class_id + 1, dtype=np.int32)
    rooms = np.zeros(dataset.max_class_id + 1, dtype=np.int32)
    
    for class_id, gene in chromosome.items():
        s = gene["start_slot"]
        r = gene["room_id"]
        starts[class_id] = s
        ends[class_id] = s + dataset.class_lengths[class_id]
        rooms[class_id] = r
            
    # --- 1. STUDENT PENALTY ---
    s1 = starts[dataset.conflict_c1]
    e1 = ends[dataset.conflict_c1]
    s2 = starts[dataset.conflict_c2]
    e2 = ends[dataset.conflict_c2]
    
    overlaps = (s1 < e2) & (s2 < e1)
    student_errors = int(np.sum(dataset.conflict_shared[overlaps]))

    # --- 2. ROOM PENALTY ---
    # A) Room overlap conflicts: group slots by room
    classes_by_room = {}
    for class_id in dataset.class_ids:
        s = starts[class_id]
        e = ends[class_id]
        r_id = rooms[class_id]
        classes_by_room.setdefault(r_id, []).append((s, e))
        
    for r_id, r_classes in classes_by_room.items():
        n = len(r_classes)
        if n < 2:
            continue
        # Sort by start slot
        r_classes.sort(key=lambda x: x[0])
        # O(N log N) check
        for i in range(n):
            s1, e1 = r_classes[i]
            for j in range(i + 1, n):
                s2, e2 = r_classes[j]
                if s2 >= e1:
                    break
                room_overlap_errors += 1
                    
    # B) Capacity shortages (100% Vectorized)
    assigned_rooms = rooms[dataset.class_ids]
    caps = dataset.room_capacities[assigned_rooms]
    diff = dataset.class_enrollments[dataset.class_ids] - caps
    capacity_errors = int(np.sum(diff[diff > 0]))
            
    # C) Travel distance for consecutive classes (Fast Matrix Lookups)
    GAP_THRESHOLD = 36  # 3 hours (36 * 5 mins)
    for student_id, st_classes in dataset.student_classes.items():
        if len(st_classes) < 2:
            continue
            
        # Dynamically sort using starts lookup in compiled C
        sorted_classes = sorted(st_classes, key=starts.__getitem__)
        
        for i in range(len(sorted_classes) - 1):
            c1 = sorted_classes[i]
            c2 = sorted_classes[i+1]
            s1 = starts[c1]
            s2 = starts[c2]
            
            gap = s2 - ends[c1]
            day_1 = s1 // dataset.slots_per_day
            day_2 = s2 // dataset.slots_per_day
            
            if day_1 == day_2 and 0 <= gap <= GAP_THRESHOLD:
                travel_errors += dataset.travel_distances_matrix[rooms[c1], rooms[c2]]
                
    room_errors = room_overlap_errors + capacity_errors + travel_errors


    # --- 3. TIME PENALTY ---
    # Instantly resolved via precomputed set lookups
    for class_id in dataset.class_ids:
        s_slot = starts[class_id]
        allowed_set = dataset.allowed_slots[class_id]
        if allowed_set is not None:
            if s_slot not in allowed_set:
                time_errors += 1

    # --- 4. DISTRIBUTION PENALTY ---
    for dist in dataset.distributions:
        classes = dist.classes
        if len(classes) < 2:
            continue
            
        c_type = dist.type
        if c_type == "Precedence":
            for i in range(len(classes) - 1):
                c1, c2 = classes[i], classes[i+1]
                if ends[c1] > starts[c2]:
                    dist_errors += 1
                    
        elif c_type == "SameTime":
            first_slot = starts[classes[0]] % dataset.slots_per_day
            for c in classes[1:]:
                if (starts[c] % dataset.slots_per_day) != first_slot:
                    dist_errors += 1
                    
        elif c_type == "SameDays":
            first_day = starts[classes[0]] // dataset.slots_per_day
            for c in classes[1:]:
                if (starts[c] // dataset.slots_per_day) != first_day:
                    dist_errors += 1
                    
        elif c_type == "DifferentDays":
            days = [starts[c] // dataset.slots_per_day for c in classes]
            if len(days) != len(set(days)):
                dist_errors += (len(days) - len(set(days)))
                
        elif c_type == "SameRoom":
            first_room = rooms[classes[0]]
            for c in classes[1:]:
                if rooms[c] != first_room:
                    dist_errors += 1
                    
        elif c_type == "SameWeeks":
            first_week = starts[classes[0]] // (7 * dataset.slots_per_day)
            for c in classes[1:]:
                if (starts[c] // (7 * dataset.slots_per_day)) != first_week:
                    dist_errors += 1
                    
        elif c_type in {"NotOverlap", "SameAttendees"}:
            for i, c1 in enumerate(classes):
                s1 = starts[c1]
                e1 = ends[c1]
                for c2 in classes[i+1:]:
                    if s1 < ends[c2] and starts[c2] < e1:
                        dist_errors += 1

    # --- 5. TOTAL PENALTY & FITNESS ---
    student_weight = dataset.weights.get("student", 5.0)
    room_weight = dataset.weights.get("room", 1.0)
    time_weight = dataset.weights.get("time", 4.0)
    dist_weight = dataset.weights.get("distribution", 15.0)
    
    total_penalty = (
        (student_errors * student_weight) +
        (room_errors * room_weight) +
        (time_errors * time_weight) +
        (dist_errors * dist_weight)
    )
    
    fitness = float(total_penalty)
    
    scores = {
        "student": int(student_errors),
        "room": int(room_errors),
        "time": int(time_errors),
        "distribution": int(dist_errors),
        "total_penalty": int(total_penalty)
    }
    
    return fitness, scores
