from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np


@dataclass(frozen=True)
class Classroom:
    id: int
    capacity: int
    building_name: str
    room_number: str
    room_type: str = ""


@dataclass(frozen=True)
class Course:
    id: int
    name: str = ""
    department: str = ""
    credits: int = 0
    description: str = ""


@dataclass(frozen=True)
class Instructor:
    id: str
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    phone_number: str = ""
    department: str = ""


@dataclass(frozen=True)
class TimeSlot:
    id: int
    day: str
    start_time: str
    end_time: str


@dataclass
class CSVDataset:
    classrooms: Dict[int, Classroom]
    courses: Dict[int, Course]
    instructors: Dict[str, Instructor]
    timeslots: Dict[int, TimeSlot]
    students: Dict[str, Dict[str, Any]]
    schedule_records: List[Dict[str, Any]]

    course_ids: List[int]
    course_id_to_index: Dict[int, int]
    index_to_course_id: List[int]

    course_enrollment_counts: Dict[int, int]
    course_students: Dict[int, List[str]]
    student_courses: Dict[str, List[int]]
    course_instructors: Dict[int, List[str]]
    conflict_matrix: np.ndarray
