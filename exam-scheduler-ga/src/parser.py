import xml.etree.ElementTree as ET
import numpy as np
import logging
from .model import Period, Room, Exam, Dataset

logger = logging.getLogger(__name__)

def parse_dataset(xml_path: str) -> Dataset:
    logger.info(f"Parsing dataset from {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    dataset = Dataset()
    
    # 1. Parse Periods
    periods_elem = root.find('periods')
    if periods_elem is not None:
        for p_elem in periods_elem.findall('period'):
            p_id = int(p_elem.get('id'))
            dataset.periods[p_id] = Period(
                id=p_id,
                length=int(p_elem.get('length')),
                day=p_elem.get('day'),
                time=p_elem.get('time'),
                penalty=int(p_elem.get('penalty'))
            )
            
    # 2. Parse Rooms
    rooms_elem = root.find('rooms')
    if rooms_elem is not None:
        for r_elem in rooms_elem.findall('room'):
            r_id = int(r_elem.get('id'))
            coords = r_elem.get('coordinates', '')
            lat, lon = 0.0, 0.0
            if coords and ',' in coords:
                lat, lon = map(float, coords.split(','))
            
            room = Room(
                id=r_id,
                size=int(r_elem.get('size')),
                alt_size=int(r_elem.get('alt')),
                lat=lat,
                lon=lon
            )
            for p_elem in r_elem.findall('period'):
                p_id = int(p_elem.get('id'))
                if p_elem.get('available') == 'false':
                    room.unavailable_periods.add(p_id)
                elif p_elem.get('penalty') is not None:
                    room.penalty_periods[p_id] = int(p_elem.get('penalty'))
            dataset.rooms[r_id] = room
            
    # 3. Parse Exams (without student counts yet)
    exams_elem = root.find('exams')
    if exams_elem is not None:
        all_exams = exams_elem.findall('exam')
        # Sort by id to ensure deterministic indexing
        all_exams.sort(key=lambda x: int(x.get('id')))
        
        for i, e_elem in enumerate(all_exams):
            e_id = int(e_elem.get('id'))
            exam = Exam(
                id=e_id,
                idx=i,  # 0 to 2222
                length=int(e_elem.get('length')),
                alt_seating=(e_elem.get('alt') == 'true')
            )
            
            for p_elem in e_elem.findall('period'):
                exam.available_periods.add(int(p_elem.get('id')))
            for r_elem in e_elem.findall('room'):
                exam.available_rooms.add(int(r_elem.get('id')))
                
            dataset.exams.append(exam)
            
    exam_id_to_idx = {exam.id: exam.idx for exam in dataset.exams}
            
    # 4. Parse Students
    students_elem = root.find('students')
    if students_elem is not None:
        for s_elem in students_elem.findall('student'):
            s_id = int(s_elem.get('id'))
            exam_ids = [int(e.get('id')) for e in s_elem.findall('exam')]
            exam_indices = [exam_id_to_idx[eid] for eid in exam_ids]
            dataset.student_exams[s_id] = exam_indices
            
            # Increment student counts for exams
            for idx in exam_indices:
                dataset.exams[idx].students_count += 1
                
    # 5. Populate derived metrics
    num_exams = len(dataset.exams)
    dataset.conflict_matrix = np.zeros((num_exams, num_exams), dtype=np.int32)
    
    for s_id, exam_indices in dataset.student_exams.items():
        for i in range(len(exam_indices)):
            for j in range(i + 1, len(exam_indices)):
                ex1, ex2 = exam_indices[i], exam_indices[j]
                dataset.conflict_matrix[ex1][ex2] += 1
                dataset.conflict_matrix[ex2][ex1] += 1
                
    for exam in dataset.exams:
        if exam.students_count >= 600:
            exam.is_large = True
            dataset.large_exams_indices.append(exam.idx)
            
        if len(exam.available_periods) == 1:
            dataset.time_fixed_exams[exam.idx] = list(exam.available_periods)[0]
            
        if len(exam.available_rooms) == 1:
            dataset.room_fixed_exams[exam.idx] = list(exam.available_rooms)[0]
            
    # 6. Parse Constraints
    constraints_elem = root.find('constraints')
    if constraints_elem is not None:
        for dp_elem in constraints_elem.findall('different-period'):
            exam_indices = [exam_id_to_idx[int(e.get('id'))] for e in dp_elem.findall('exam')]
            dataset.different_period_constraints.append(exam_indices)
            
        for sp_elem in constraints_elem.findall('same-period'):
            exam_indices = [exam_id_to_idx[int(e.get('id'))] for e in sp_elem.findall('exam')]
            dataset.same_period_constraints.append(exam_indices)
            
    return dataset

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ds = parse_dataset(r"C:/exam-timetabling-with-GA/pu-exam-fal10.xml")
    print(f"Loaded {len(ds.exams)} exams, {len(ds.rooms)} rooms, {len(ds.periods)} periods, {len(ds.student_exams)} students.")
    print(f"Large exams: {len(ds.large_exams_indices)}")
    print(f"Time-fixed exams: {len(ds.time_fixed_exams)}")
    print(f"Room-fixed exams: {len(ds.room_fixed_exams)}")
    print(f"Different period constraints: {len(ds.different_period_constraints)}")
    print(f"Same period constraints: {len(ds.same_period_constraints)}")
    
    # Verify density
    total_conflicts = np.sum(ds.conflict_matrix > 0)
    density = total_conflicts / (len(ds.exams) * (len(ds.exams) - 1))
    print(f"Conflict Density: {density * 100:.3f}%")
