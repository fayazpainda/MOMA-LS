"""
Wei2021 Reproduction

Taillard Benchmark Reader

Supports:

1. Pair format:
   machine time machine time ...

2. Standard Taillard format:
   processing-time matrix
   machine-order matrix

Taillard machine IDs are 1-based in the source files
and are converted to 0-based Python indices.
"""


from dataclasses import dataclass
from typing import List



# ============================================================
# Data Structures
# ============================================================

@dataclass
class Operation:

    machine: int

    processing_time: int



@dataclass
class Job:

    job_id: int

    operations: List[Operation]



@dataclass
class JSPInstance:

    jobs: List[Job]

    num_jobs: int

    num_machines: int



# ============================================================
# Taillard Loader
# ============================================================

def load_taillard(path):


    with open(

        path,

        "r"

    ) as file:


        lines = [

            line.strip()

            for line in file.readlines()

            if line.strip()

        ]



    if not lines:

        raise ValueError(

            f"Empty benchmark file: {path}"

        )



    # ========================================================
    # Header
    # ========================================================

    first = lines[0].split()


    if len(first) < 2:

        raise ValueError(

            "First line must contain "
            "number of jobs and machines."

        )


    num_jobs = int(

        first[0]

    )


    num_machines = int(

        first[1]

    )



    if num_jobs <= 0 or num_machines <= 0:

        raise ValueError(

            "Number of jobs and machines "
            "must be positive."

        )



    data_lines = lines[1:]


    if not data_lines:

        raise ValueError(

            "Benchmark contains no instance data."

        )



    values_per_line = len(

        data_lines[0].split()

    )


    jobs = []



    # ========================================================
    # Format 1:
    # machine time machine time ...
    # ========================================================

    if values_per_line == (

        num_machines * 2

    ):


        if len(data_lines) < num_jobs:

            raise ValueError(

                "Not enough job rows in pair-format file."

            )


        for job_id in range(

            num_jobs

        ):


            data = list(

                map(

                    int,

                    data_lines[
                        job_id
                    ].split()

                )

            )



            if len(data) != (

                num_machines * 2

            ):

                raise ValueError(

                    f"Job {job_id} row has "
                    f"{len(data)} values; expected "
                    f"{num_machines * 2}."

                )



            operations = []



            for operation_index in range(

                num_machines

            ):


                machine = data[

                    operation_index * 2

                ]


                processing_time = data[

                    operation_index * 2 + 1

                ]



                # ------------------------------------------------
                # Pair-format files may be 0-based or 1-based.
                # Normalize if clearly 1-based.
                # ------------------------------------------------

                if (

                    1 <= machine <= num_machines

                ):

                    normalized_machine = (

                        machine - 1

                    )

                else:

                    normalized_machine = (

                        machine

                    )



                if not (

                    0
                    <=
                    normalized_machine
                    <
                    num_machines

                ):

                    raise ValueError(

                        f"Invalid machine ID "
                        f"{machine} for job {job_id}."

                    )



                if processing_time <= 0:

                    raise ValueError(

                        "Processing times must be positive."

                    )



                operations.append(

                    Operation(

                        machine=
                            normalized_machine,

                        processing_time=
                            processing_time

                    )

                )



            jobs.append(

                Job(

                    job_id=
                        job_id,

                    operations=
                        operations

                )

            )



    # ========================================================
    # Format 2:
    # Standard Taillard
    #
    # First n rows:
    # processing-time matrix
    #
    # Next n rows:
    # machine-order matrix
    # ========================================================

    else:


        expected_rows = (

            2 * num_jobs

        )


        if len(data_lines) < expected_rows:

            raise ValueError(

                "Standard Taillard file does not contain "
                f"the required {expected_rows} data rows."

            )



        processing_matrix = []

        machine_matrix = []



        # ----------------------------------------------------
        # Processing-time matrix
        # ----------------------------------------------------

        for job_id in range(

            num_jobs

        ):


            row = list(

                map(

                    int,

                    data_lines[
                        job_id
                    ].split()

                )

            )



            if len(row) != num_machines:

                raise ValueError(

                    f"Processing row {job_id} has "
                    f"{len(row)} values; expected "
                    f"{num_machines}."

                )



            if any(

                processing_time <= 0

                for processing_time
                in row

            ):

                raise ValueError(

                    "Processing times must be positive."

                )



            processing_matrix.append(

                row

            )



        # ----------------------------------------------------
        # Machine-order matrix
        # ----------------------------------------------------

        start = num_jobs


        for job_id in range(

            num_jobs

        ):


            row = list(

                map(

                    int,

                    data_lines[
                        start + job_id
                    ].split()

                )

            )



            if len(row) != num_machines:

                raise ValueError(

                    f"Machine row {job_id} has "
                    f"{len(row)} values; expected "
                    f"{num_machines}."

                )



            # ------------------------------------------------
            # Standard Taillard machine IDs are 1...m
            # ------------------------------------------------

            if any(

                machine < 1
                or
                machine > num_machines

                for machine
                in row

            ):

                raise ValueError(

                    f"Invalid machine ID in job {job_id}."

                )



            machine_matrix.append(

                row

            )



        # ====================================================
        # Build Jobs
        # ====================================================

        for job_id in range(

            num_jobs

        ):


            operations = []



            for operation_index in range(

                num_machines

            ):


                # ------------------------------------------------
                # Taillard:
                # machine IDs are 1-based.
                #
                # Python:
                # convert to 0-based.
                # ------------------------------------------------

                machine = (

                    machine_matrix[
                        job_id
                    ][
                        operation_index
                    ]

                    -

                    1

                )


                processing_time = (

                    processing_matrix[
                        job_id
                    ][
                        operation_index
                    ]

                )



                operations.append(

                    Operation(

                        machine=
                            machine,

                        processing_time=
                            processing_time

                    )

                )



            jobs.append(

                Job(

                    job_id=
                        job_id,

                    operations=
                        operations

                )

            )



    # ========================================================
    # Final Instance
    # ========================================================

    return JSPInstance(

        jobs=
            jobs,

        num_jobs=
            num_jobs,

        num_machines=
            num_machines

    )



# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":

    print(

        "Taillard benchmark reader loaded successfully"

    )