import xml.etree.ElementTree as ET
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

class Room:
    def __init__(self, room_id: int, capacity: int, travel_distances: Dict[int, int]):
        self.id = room_id
        self.capacity = capacity
        self.travel_distances = travel_distances  # target_room_id -> travel_value

class Student:
    def __init__(self, student_id: int, courses: List[int]):
        self.id = student_id
        self.courses = courses  # list of course_ids

class DistributionConstraint:
    def __init__(self, constraint_type: str, classes: List[int]):
        self.type = constraint_type
        self.classes = classes  # list of class_ids affected by this constraint

class ClassTime:
    def __init__(self, days: str, start: int, length: int, weeks: str, penalty: int):
        self.days = days          # e.g., "1000000" (Monday)
        self.start = start        # start slot in day (0-287)
        self.length = length      # duration in slots
        self.weeks = weeks        # e.g., "011110000000000000"
        self.penalty = penalty

class ClassInfo:
    def __init__(self, class_id: int, course_id: int, limit: int, allowed_times: List[ClassTime], allowed_rooms: List[int]):
        self.id = class_id
        self.course_id = course_id
        self.limit = limit
        self.allowed_times = allowed_times
        self.allowed_rooms = allowed_rooms
        
        # Determine fixed length from allowed times, default to 24 slots (2 hours) if none
        self.length = allowed_times[0].length if allowed_times else 24

class XMLDataset:
    def __init__(
        self,
        name: str,
        nr_days: int,
        nr_weeks: int,
        slots_per_day: int,
        weights: Dict[str, float],
        rooms: Dict[int, Room],
        classes: Dict[int, ClassInfo],
        students: Dict[int, Student],
        distributions: List[DistributionConstraint]
    ):
        self.name = name
        self.nr_days = nr_days
        self.nr_weeks = nr_weeks
        self.slots_per_day = slots_per_day
        self.total_slots = nr_weeks * nr_days * slots_per_day
        
        self.weights = weights
        self.rooms = rooms
        self.classes = classes
        self.students = students
        self.distributions = distributions
        
        # Optimization lookups
        self.class_ids = sorted(classes.keys())
        self.room_ids = sorted(rooms.keys())
        
        # Map course_id -> list of class_ids
        self.course_classes: Dict[int, List[int]] = {}
        for class_id, cl in classes.items():
            self.course_classes.setdefault(cl.course_id, []).append(class_id)
            
        # Map course_id -> list of student_ids
        self.course_students: Dict[int, List[int]] = {}
        for student_id, st in students.items():
            for course_id in st.courses:
                self.course_students.setdefault(course_id, []).append(student_id)
                
        # Map class_id -> student enrollment count
        self.class_enrollment: Dict[int, int] = {}
        for class_id, cl in classes.items():
            self.class_enrollment[class_id] = len(self.course_students.get(cl.course_id, []))
            
        # Map student_id -> list of class_ids they must attend
        self.student_classes: Dict[int, List[int]] = {}
        for student_id, st in students.items():
            student_cls = []
            for course_id in st.courses:
                if course_id in self.course_classes:
                    student_cls.extend(self.course_classes[course_id])
            self.student_classes[student_id] = sorted(list(set(student_cls)))
            
        # Precompute conflicting class pairs sharing students
        logger.info("Precomputing conflicting class pairs...")
        pair_counter = {}
        for student_id, st_cls in self.student_classes.items():
            for i, c1 in enumerate(st_cls):
                for c2 in st_cls[i+1:]:
                    pair = (c1, c2) if c1 < c2 else (c2, c1)
                    pair_counter[pair] = pair_counter.get(pair, 0) + 1
                    
        self.conflict_pairs = [(pair[0], pair[1], count) for pair, count in pair_counter.items() if count > 0]
        
        # NumPy advanced indexing arrays for vectorization
        self.conflict_c1 = np.array([p[0] for p in self.conflict_pairs], dtype=np.int32)
        self.conflict_c2 = np.array([p[1] for p in self.conflict_pairs], dtype=np.int32)
        self.conflict_shared = np.array([p[2] for p in self.conflict_pairs], dtype=np.int32)
        
        logger.info(f"Precomputed {len(self.conflict_pairs)} conflicting class pairs sharing students.")
        
        # --- PRECOMPUTE HIGH SPEED LOOKUPS ---
        self.max_class_id = max(self.class_ids) if self.class_ids else 0
        self.max_room_id = max(self.room_ids) if self.room_ids else 0
        
        # 1. Precompute class lengths array
        self.class_lengths = np.zeros(self.max_class_id + 1, dtype=np.int32)
        for class_id, cl in self.classes.items():
            self.class_lengths[class_id] = cl.length
            
        # 2. Precompute room capacities array
        self.room_capacities = np.zeros(self.max_room_id + 1, dtype=np.int32)
        for room_id, r in self.rooms.items():
            self.room_capacities[room_id] = r.capacity
            
        # 3. Precompute class enrollments array
        self.class_enrollments = np.zeros(self.max_class_id + 1, dtype=np.int32)
        for class_id in self.class_ids:
            self.class_enrollments[class_id] = self.class_enrollment[class_id]
            
        # 4. Precompute allowed slots set mapping
        self.allowed_slots = {}
        for class_id, cl in self.classes.items():
            if not cl.allowed_times:
                self.allowed_slots[class_id] = None  # any is allowed
                continue
                
            allowed_set = set()
            for allowed in cl.allowed_times:
                for w in range(self.nr_weeks):
                    if w < len(allowed.weeks) and allowed.weeks[w] == '1':
                        for d in range(7):
                            if d < len(allowed.days) and allowed.days[d] == '1':
                                slot = w * (7 * self.slots_per_day) + d * self.slots_per_day + allowed.start
                                allowed_set.add(slot)
            self.allowed_slots[class_id] = allowed_set
            
        # 5. Precompute room travel distances matrix
        self.travel_distances_matrix = np.zeros((self.max_room_id + 1, self.max_room_id + 1), dtype=np.int32)
        for room_id, r in self.rooms.items():
            for target_id, dist in r.travel_distances.items():
                self.travel_distances_matrix[room_id, target_id] = dist


def parse_xml_dataset(xml_path: str) -> XMLDataset:
    logger.info(f"Parsing XML dataset from {xml_path}...")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Root attributes
    name = root.get("name", "unnamed")
    nr_days = int(root.get("nrDays", "7"))
    nr_weeks = int(root.get("nrWeeks", "18"))
    slots_per_day = int(root.get("slotsPerDay", "288"))
    
    # 1. Parse optimization weights
    opt_elem = root.find("optimization")
    weights = {
        "time": 4.0,
        "room": 1.0,
        "distribution": 15.0,
        "student": 5.0
    }
    if opt_elem is not None:
        for w_name in weights.keys():
            if opt_elem.get(w_name) is not None:
                weights[w_name] = float(opt_elem.get(w_name))
    logger.info(f"Optimization weights: {weights}")
    
    # 2. Parse rooms
    rooms = {}
    rooms_elem = root.find("rooms")
    if rooms_elem is not None:
        for r_elem in rooms_elem.findall("room"):
            room_id = int(r_elem.get("id"))
            capacity = int(r_elem.get("capacity") or r_elem.get("size") or "10")
            
            # travel distances
            travel_distances = {}
            for t_elem in r_elem.findall("travel"):
                target_room = int(t_elem.get("room"))
                val = int(t_elem.get("value"))
                travel_distances[target_room] = val
                
            rooms[room_id] = Room(room_id, capacity, travel_distances)
    logger.info(f"Parsed {len(rooms)} rooms.")
    
    # 3. Parse courses and classes
    classes = {}
    courses_elem = root.find("courses")
    if courses_elem is not None:
        for c_elem in courses_elem.findall("course"):
            course_id = int(c_elem.get("id"))
            
            for config_elem in c_elem.findall("config"):
                for subpart_elem in config_elem.findall("subpart"):
                    for cl_elem in subpart_elem.findall("class"):
                        class_id = int(cl_elem.get("id"))
                        limit = int(cl_elem.get("limit", "0"))
                        
                        # allowed times
                        allowed_times = []
                        for t_elem in cl_elem.findall("time"):
                            allowed_times.append(ClassTime(
                                days=t_elem.get("days"),
                                start=int(t_elem.get("start")),
                                length=int(t_elem.get("length")),
                                weeks=t_elem.get("weeks"),
                                penalty=int(t_elem.get("penalty", "0"))
                            ))
                            
                        # allowed rooms
                        allowed_rooms = []
                        for r_elem in cl_elem.findall("room"):
                            allowed_rooms.append(int(r_elem.get("id")))
                            
                        classes[class_id] = ClassInfo(
                            class_id=class_id,
                            course_id=course_id,
                            limit=limit,
                            allowed_times=allowed_times,
                            allowed_rooms=allowed_rooms
                        )
    logger.info(f"Parsed {len(classes)} classes.")
    
    # 4. Parse distributions
    distributions = []
    dist_elem = root.find("distributions")
    if dist_elem is not None:
        for d_elem in dist_elem.findall("distribution"):
            dist_type = d_elem.get("type")
            affected_classes = [int(c.get("id")) for c in d_elem.findall("class")]
            distributions.append(DistributionConstraint(dist_type, affected_classes))
    logger.info(f"Parsed {len(distributions)} distribution constraints.")
    
    # 5. Parse students
    students = {}
    stud_elem = root.find("students")
    if stud_elem is not None:
        for s_elem in stud_elem.findall("student"):
            student_id = int(s_elem.get("id"))
            courses_registered = [int(c.get("id")) for c in s_elem.findall("course")]
            students[student_id] = Student(student_id, courses_registered)
    logger.info(f"Parsed {len(students)} students.")
    
    return XMLDataset(
        name=name,
        nr_days=nr_days,
        nr_weeks=nr_weeks,
        slots_per_day=slots_per_day,
        weights=weights,
        rooms=rooms,
        classes=classes,
        students=students,
        distributions=distributions
    )
