from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple
import numpy as np

@dataclass
class Period:
    id: int
    length: int
    day: str
    time: str
    penalty: int

@dataclass
class Room:
    id: int
    size: int
    alt_size: int
    lat: float = 0.0
    lon: float = 0.0
    penalty_periods: Dict[int, int] = field(default_factory=dict)
    unavailable_periods: Set[int] = field(default_factory=set)

@dataclass
class Exam:
    id: int
    idx: int  # 0-based index for arrays
    length: int
    alt_seating: bool
    students_count: int = 0
    available_periods: Set[int] = field(default_factory=set)
    available_rooms: Set[int] = field(default_factory=set)
    is_large: bool = False

class Dataset:
    def __init__(self):
        self.periods: Dict[int, Period] = {}
        self.rooms: Dict[int, Room] = {}
        self.exams: List[Exam] = [] # index corresponds to exam.idx
        self.student_exams: Dict[int, List[int]] = {} # student_id -> list of exam_idx
        
        self.conflict_matrix: np.ndarray = None # 2223x2223 numpy array
        
        # Hard Constraints (Distribution constraints)
        self.different_period_constraints: List[List[int]] = [] # list of exam_idx lists
        self.same_period_constraints: List[List[int]] = [] # list of exam_idx lists
        
        # Pre-calculated arrays for fast access
        self.large_exams_indices: List[int] = []
        self.time_fixed_exams: Dict[int, int] = {} # exam_idx -> period_id
        self.room_fixed_exams: Dict[int, int] = {} # exam_idx -> room_id
