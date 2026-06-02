from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .model import CSVDataset

Gene = Dict[str, object]
Chromosome = Dict[int, Gene]


class FitnessCalculator:
    def __init__(self, dataset: CSVDataset):
        self.ds = dataset
        self.weights = {
            "instructor_conflict": 1000,
            "room_conflict": 1000,
            "student_conflict": 1000,
            "capacity_shortage": 500,
            "building_spread": 100,
            "same_day_extra_exam": 50,
            "back_to_back_exam": 150,
        }
        self._timeslot_day = {
            timeslot_id: timeslot.day
            for timeslot_id, timeslot in self.ds.timeslots.items()
        }
        self._consecutive_timeslots = self._build_consecutive_timeslot_pairs()

    def _build_consecutive_timeslot_pairs(self) -> set[frozenset[int]]:
        timeslots_by_day = defaultdict(list)
        for timeslot in self.ds.timeslots.values():
            timeslots_by_day[timeslot.day].append(timeslot)

        pairs = set()
        for day_timeslots in timeslots_by_day.values():
            ordered = sorted(day_timeslots, key=lambda slot: (slot.start_time, slot.end_time, slot.id))
            for previous, current in zip(ordered, ordered[1:]):
                pairs.add(frozenset((previous.id, current.id)))
        return pairs

    def _validate_chromosome(self, chromosome: Chromosome) -> None:
        missing_courses = [course_id for course_id in self.ds.course_ids if course_id not in chromosome]
        if missing_courses:
            raise ValueError(f"Chromosome is missing course_id values: {missing_courses}")

        unknown_courses = sorted(set(chromosome) - set(self.ds.course_ids))
        if unknown_courses:
            raise ValueError(f"Chromosome references unknown course_id values: {unknown_courses}")

        for course_id, gene in chromosome.items():
            timeslot_id = int(gene["timeslot"])
            if timeslot_id not in self.ds.timeslots:
                raise ValueError(f"Course {course_id} references unknown timeslot_id: {timeslot_id}")

            room_ids = list(gene["rooms"])
            unknown_rooms = sorted(set(room_ids) - set(self.ds.classrooms))
            if unknown_rooms:
                raise ValueError(f"Course {course_id} references unknown classroom_id values: {unknown_rooms}")

    def _courses_by_timeslot(self, chromosome: Chromosome) -> Dict[int, List[int]]:
        courses_by_timeslot = defaultdict(list)
        for course_id, gene in chromosome.items():
            courses_by_timeslot[int(gene["timeslot"])].append(course_id)
        return courses_by_timeslot

    def _score_instructor_conflicts(self, courses_by_timeslot: Dict[int, List[int]]) -> int:
        penalty = 0
        for course_ids in courses_by_timeslot.values():
            instructor_usage = Counter()
            for course_id in course_ids:
                instructor_usage.update(self.ds.course_instructors[course_id])

            for usage_count in instructor_usage.values():
                if usage_count > 1:
                    penalty += (usage_count * (usage_count - 1) // 2) * self.weights["instructor_conflict"]
        return penalty

    def _score_room_conflicts(self, chromosome: Chromosome) -> int:
        room_usage_by_timeslot = defaultdict(Counter)
        for gene in chromosome.values():
            timeslot_id = int(gene["timeslot"])
            room_ids = list(gene["rooms"])
            room_usage_by_timeslot[timeslot_id].update(room_ids)

        penalty = 0
        for room_usage in room_usage_by_timeslot.values():
            for usage_count in room_usage.values():
                if usage_count > 1:
                    penalty += (usage_count * (usage_count - 1) // 2) * self.weights["room_conflict"]
        return penalty

    def _score_student_conflicts(self, courses_by_timeslot: Dict[int, List[int]]) -> int:
        penalty = 0
        for course_ids in courses_by_timeslot.values():
            if len(course_ids) < 2:
                continue

            course_indices = [self.ds.course_id_to_index[course_id] for course_id in course_ids]
            shared_student_count = int(
                self.ds.conflict_matrix[np.ix_(course_indices, course_indices)].sum() // 2
            )
            penalty += shared_student_count * self.weights["student_conflict"]
        return penalty

    def _score_capacity_shortage(self, chromosome: Chromosome) -> int:
        penalty = 0
        for course_id, gene in chromosome.items():
            total_capacity = sum(
                self.ds.classrooms[room_id].capacity
                for room_id in set(gene["rooms"])
            )
            shortage = max(0, self.ds.course_enrollment_counts[course_id] - total_capacity)
            penalty += shortage * self.weights["capacity_shortage"]
        return penalty

    def _score_building_spread(self, chromosome: Chromosome) -> int:
        penalty = 0
        for gene in chromosome.values():
            buildings = {
                self.ds.classrooms[room_id].building_name
                for room_id in set(gene["rooms"])
            }
            if len(buildings) > 1:
                penalty += (len(buildings) - 1) * self.weights["building_spread"]
        return penalty

    def _score_student_spread(self, chromosome: Chromosome) -> Dict[str, int]:
        scores = {
            "same_day_extra_exam": 0,
            "back_to_back_exam": 0,
        }

        for course_ids in self.ds.student_courses.values():
            timeslot_ids = [int(chromosome[course_id]["timeslot"]) for course_id in course_ids]
            day_counts = Counter(self._timeslot_day[timeslot_id] for timeslot_id in timeslot_ids)

            for exam_count in day_counts.values():
                if exam_count > 1:
                    scores["same_day_extra_exam"] += (
                        exam_count - 1
                    ) * self.weights["same_day_extra_exam"]

            for left_pos, left_timeslot_id in enumerate(timeslot_ids):
                for right_timeslot_id in timeslot_ids[left_pos + 1:]:
                    if frozenset((left_timeslot_id, right_timeslot_id)) in self._consecutive_timeslots:
                        scores["back_to_back_exam"] += self.weights["back_to_back_exam"]

        return scores

    def calculate_fitness(self, chromosome: Chromosome) -> Tuple[float, Dict[str, int]]:
        self._validate_chromosome(chromosome)
        courses_by_timeslot = self._courses_by_timeslot(chromosome)

        scores = {
            "instructor_conflict": self._score_instructor_conflicts(courses_by_timeslot),
            "room_conflict": self._score_room_conflicts(chromosome),
            "student_conflict": self._score_student_conflicts(courses_by_timeslot),
            "capacity_shortage": self._score_capacity_shortage(chromosome),
            "building_spread": self._score_building_spread(chromosome),
        }
        scores.update(self._score_student_spread(chromosome))

        penalty_score = sum(scores.values())
        scores["total_penalty"] = penalty_score
        fitness_score = penalty_score
        return fitness_score, scores


def calculate_fitness(dataset: CSVDataset, chromosome: Chromosome) -> Tuple[float, Dict[str, int]]:
    return FitnessCalculator(dataset).calculate_fitness(chromosome)
