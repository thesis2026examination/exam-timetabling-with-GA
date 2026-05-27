# PROJECT CONTEXT: EXAM TIMETABLING USING GENETIC ALGORITHM (TARGET: FALL 2010 DATASET)

This file contains the specific mathematical model, constraints, penalty weights, and target benchmark results for our Genetic Algorithm (GA) based Exam Timetabling Project, configured explicitly for the **Fall 2010** dataset. All Python code and algorithmic designs to be generated must strictly adhere to the parameters outlined below.

---

## 1. FALL 2010 DATASET CHARACTERISTICS
* **Number of Exams (Exams):** Exactly **2,223** exams (genes) to be scheduled.
  * **Exams with Exam Seating:** **875** exams require specific anti-cheating seating arrangements.
* **Number of Students (Students):** Exactly **34,418** unique students.
* **Total Enrollments (Enrollments):** Exactly **129,090** course registrations.
* **Conflict Density:** **2.872%** (This represents the density of the student conflict matrix).
* **Hard Distribution Constraints:** **6** specific distribution rules that cannot be violated.
* **Pre-assigned Exams (Fixed):** **121** exams are fixed in time, and **89** exams are fixed in a specific room. These must be locked during initialization and skipped during mutation/crossover.
* **Large Exams (600+ Students):** Exactly **21** massive exams.
* **Exams Needing a Room Split:** **7** exams cannot fit into a single room and must be split into multiple rooms.
* **Time Slots (Periods):** A total of **29 periods** are available (Monday to Saturday, 5 slots of 2 hours each day: 08:00, 10:30, 13:00, 15:30, 19:00. Saturday's 19:00 slot does not exist).

---

## 2. BENCHMARK HEURISTIC RULES (FALL 2010 SPECIFIC)
1. **Large Exams Routing:** The **21 large exams** have a conflict density of over 80% among themselves. They **must strictly fit within the first 24 periods** (excluding Friday evening and Saturday slots).
2. **Seating Capacity:** For the **875 exams** requiring exam seating, the available room capacity must be halved (or custom-diluted based on dataset rules) to prevent students from sitting next to each other.
3. **Room Splitting Framework:** The algorithm must support assigning the **7 split-heavy exams** to at least two different rooms during the exact same period seamlessly.

---

## 3. FITNESS FUNCTION PENALTY WEIGHTS (PRODUCTION CONFIG)
The goal of the algorithm is to **minimize** the total penalty score. The fitness function in Python must apply the following specific weights from the MISTA 2013 `production` config:

| Constraint / Violation Name | Penalty Weight | Description / Rule |
| :--- | :--- | :--- |
| **Direct Conflict** | **1,000** | A student scheduled for more than one exam in the same period (Severe Hard Constraint). |
| **Large Exams Weight** | **2,500,000** | Any of the 21 large exams spilling past the first 24 periods (Critical Hard Constraint). |
| **More Than 2 A Day** | **100** | A student having 3 or more exams on the same day. |
| **Back-To-Back** | **10** | A student having consecutive exams on the same day with no gap period. |
| **Room Split** | **10** | Penalty applied whenever an exam is split into multiple rooms. |
| **Period Penalty** | **1** | Penalty = 4 for Saturday periods, Penalty = 1 for Friday afternoon periods. |
| **Room Penalty** | **1** | Scheduling exams in discouraged or non-preferred rooms. |
| **Room Split Distance** | **0.01** | Distance multiplier (meters) between split rooms of the same exam. |
| **Room Size** | **0.001** | Inefficient use of room capacities. |
| **Room Distance** | **0.0001** | Walking distance multiplier between rooms for students with back-to-back exams. |
| **Rotation Penalty** | **0.0001** | Inconsistency in the rotation schedule of invigilators. |

---

## 4. TARGET PERFORMANCE BENCHMARK (EXPECTED RESULTS FOR FALL 2010)
When testing our Python Genetic Algorithm against the Fall 2010 dataset, our solution quality report must track and aim to approach the following benchmark values achieved in the literature:
* **Direct Conflicts Score:** Target around **$120.5 \pm 2.7$** or lower.
* **More Than 2 A Day Score:** Target around **$494.3 \pm 14.6$** or lower.
* **Back-To-Back Score:** Target around **$5181.5 \pm 62.4$**.
* **Large Exams Penalty Score:** **Must strictly be 0.0.**
* **Period Preference Success:** Aim for **91.1%**.
* **Room Preference Success:** Aim for **72.4%**.
* **Average Room Distance:** ~49.3 meters.

---

## 5. TECHNICAL GEREKSİNİMLER VE MİMARİ
* **Language:** Python 3.x
* **Libraries:** NumPy (for vectorizing the 2223 x 2223 student conflict matrix), Pandas (for fast dataset parsing).
* **Chromosome Representation:** An array of size 2223 where `Chromosome[Exam_ID] = [Period_ID, Room_ID]`.