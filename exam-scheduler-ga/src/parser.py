import numpy as np
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .model import (
    Classroom,
    Course,
    CSVDataset,
    Instructor,
    TimeSlot,
)

logger = logging.getLogger(__name__)


def _require_columns(frame: pd.DataFrame, file_name: str, columns: List[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{file_name} is missing required columns: {', '.join(missing)}")


def _read_csv(data_dir: Path, file_name: str) -> pd.DataFrame:
    csv_path = data_dir / file_name
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_csv_dataset(data_dir: str = "archive") -> CSVDataset:
    """Load the small CSV exam-timetabling dataset and build GA-ready lookups."""
    data_path = Path(data_dir)
    logger.info("Loading CSV dataset from %s", data_path)

    classrooms_df = _read_csv(data_path, "classrooms.csv")
    courses_df = _read_csv(data_path, "courses.csv")
    instructors_df = _read_csv(data_path, "instructors.csv")
    timeslots_df = _read_csv(data_path, "timeslots.csv")
    students_df = _read_csv(data_path, "students.csv")
    schedule_df = _read_csv(data_path, "schedule.csv")

    _require_columns(
        classrooms_df,
        "classrooms.csv",
        ["classroom_id", "capacity", "building_name", "room_number"],
    )
    _require_columns(courses_df, "courses.csv", ["course_id"])
    _require_columns(instructors_df, "instructors.csv", ["instructor_id"])
    _require_columns(
        timeslots_df,
        "timeslots.csv",
        ["timeslot_id", "day", "start_time", "end_time"],
    )
    _require_columns(students_df, "students.csv", ["student_id"])
    _require_columns(schedule_df, "schedule.csv", ["student_id", "course_id", "instructor_id"])

    classrooms_df = classrooms_df.copy()
    courses_df = courses_df.copy()
    instructors_df = instructors_df.copy()
    timeslots_df = timeslots_df.copy()
    students_df = students_df.copy()
    schedule_df = schedule_df.copy()

    for frame, columns in [
        (classrooms_df, ["classroom_id", "capacity"]),
        (courses_df, ["course_id"]),
        (timeslots_df, ["timeslot_id"]),
        (schedule_df, ["course_id"]),
    ]:
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)

    for frame, columns in [
        (classrooms_df, ["building_name", "room_number", "room_type"]),
        (courses_df, ["course_name", "department", "description"]),
        (
            instructors_df,
            ["instructor_id", "first_name", "last_name", "email", "phone_number", "department"],
        ),
        (timeslots_df, ["day", "start_time", "end_time"]),
        (students_df, ["student_id"]),
        (schedule_df, ["student_id", "instructor_id"]),
    ]:
        for column in columns:
            if column in frame.columns:
                frame[column] = frame[column].fillna("").astype(str).str.strip()

    if "credits" in courses_df.columns:
        courses_df["credits"] = pd.to_numeric(courses_df["credits"], errors="coerce").fillna(0).astype(int)

    classrooms = {
        row.classroom_id: Classroom(
            id=row.classroom_id,
            capacity=row.capacity,
            building_name=row.building_name,
            room_number=row.room_number,
            room_type=getattr(row, "room_type", ""),
        )
        for row in classrooms_df.itertuples(index=False)
    }

    courses = {
        row.course_id: Course(
            id=row.course_id,
            name=getattr(row, "course_name", ""),
            department=getattr(row, "department", ""),
            credits=getattr(row, "credits", 0),
            description=getattr(row, "description", ""),
        )
        for row in courses_df.itertuples(index=False)
    }

    instructors = {
        row.instructor_id: Instructor(
            id=row.instructor_id,
            first_name=getattr(row, "first_name", ""),
            last_name=getattr(row, "last_name", ""),
            email=getattr(row, "email", ""),
            phone_number=getattr(row, "phone_number", ""),
            department=getattr(row, "department", ""),
        )
        for row in instructors_df.itertuples(index=False)
    }

    timeslots = {
        row.timeslot_id: TimeSlot(
            id=row.timeslot_id,
            day=row.day,
            start_time=row.start_time,
            end_time=row.end_time,
        )
        for row in timeslots_df.itertuples(index=False)
    }

    students = students_df.set_index("student_id", drop=False).to_dict(orient="index")
    course_ids = sorted(courses)
    course_id_to_index = {course_id: idx for idx, course_id in enumerate(course_ids)}

    unknown_courses = sorted(set(schedule_df["course_id"]) - set(course_ids))
    if unknown_courses:
        raise ValueError(f"schedule.csv references unknown course_id values: {unknown_courses}")

    enrollment_rows = schedule_df[["student_id", "course_id"]].drop_duplicates()
    course_students: Dict[int, List[str]] = {
        course_id: sorted(
            enrollment_rows.loc[enrollment_rows["course_id"] == course_id, "student_id"].tolist()
        )
        for course_id in course_ids
    }
    course_enrollment_counts = {
        course_id: len(student_ids)
        for course_id, student_ids in course_students.items()
    }

    student_courses = {
        student_id: sorted(group["course_id"].unique().tolist())
        for student_id, group in enrollment_rows.groupby("student_id")
    }

    course_instructors = {
        course_id: sorted(
            schedule_df.loc[
                schedule_df["course_id"] == course_id,
                "instructor_id",
            ].dropna().unique().tolist()
        )
        for course_id in course_ids
    }

    incidence = pd.crosstab(enrollment_rows["student_id"], enrollment_rows["course_id"])
    incidence = incidence.reindex(columns=course_ids, fill_value=0).astype(np.int32)
    conflict_matrix = incidence.to_numpy(dtype=np.int32).T @ incidence.to_numpy(dtype=np.int32)
    np.fill_diagonal(conflict_matrix, 0)

    return CSVDataset(
        classrooms=classrooms,
        courses=courses,
        instructors=instructors,
        timeslots=timeslots,
        students=students,
        schedule_records=schedule_df.to_dict(orient="records"),
        course_ids=course_ids,
        course_id_to_index=course_id_to_index,
        index_to_course_id=course_ids,
        course_enrollment_counts=course_enrollment_counts,
        course_students=course_students,
        student_courses=student_courses,
        course_instructors=course_instructors,
        conflict_matrix=conflict_matrix,
    )
