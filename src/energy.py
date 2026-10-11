"""
Wei2021 Reproduction

Machine Energy Model

Implementation based on Wei et al.:

Eq. (1)-(2):
    Keep machine IDLE.

Eq. (3)-(4):
    Switch machine to STANDBY.

Eq. (5)-(6):
    Switch machine OFF and then ON.

Eq. (7):
    Total energy before energy-saving strategy.

Eq. (8):
    Total energy after energy-saving strategy.

Eq. (11):
    Non-processing energy consumption (NEC).

Machine states:

1. Working
2. Idle
3. Standby
4. Off

Important:
A standby/off state is treated as feasible only when the
non-processing interval is at least as long as the
corresponding ramp-up time. This prevents negative
standby duration and physically impossible state changes.
"""


from dataclasses import dataclass



# ============================================================
# Machine Energy Parameters
# ============================================================

@dataclass
class MachineEnergy:

    # P_W,j
    processing_power: float

    # P_I,j
    idle_power: float

    # P_S,j
    standby_power: float

    # P_rampup,standby,j
    ramp_standby_power: float

    # t_rampup,standby,j
    ramp_standby_time: float

    # P_rampup,off,j
    ramp_off_power: float

    # t_rampup,off,j
    ramp_off_time: float



# ============================================================
# Load Energy Parameters
# ============================================================

def load_machine_energy(
        extended_data
):

    machines = []


    for machine in extended_data["machines"]:

        machines.append(

            MachineEnergy(

                processing_power=
                    machine["processing_power"],

                idle_power=
                    machine["idle_power"],

                standby_power=
                    machine["standby_power"],

                ramp_standby_power=
                    machine["ramp_standby_power"],

                ramp_standby_time=
                    machine["ramp_standby_time"],

                ramp_off_power=
                    machine["ramp_off_power"],

                ramp_off_time=
                    machine["ramp_off_time"]

            )

        )


    return machines



# ============================================================
# Machine Operations
# ============================================================

def get_machine_operations(
        schedule,
        machine_id
):

    """
    Return operations executed on machine j
    in chronological order.
    """

    operations = [

        operation

        for operation in schedule.operations

        if operation.machine == machine_id

    ]


    operations.sort(

        key=lambda operation:
            operation.start

    )


    return operations



# ============================================================
# Non-Processing Intervals
# ============================================================

def calculate_machine_idle_periods(
        schedule,
        machine_id
):

    """
    Determine non-processing intervals over [0, Cmax].

    Intervals included:

    1. From time 0 to first operation.
    2. Between consecutive operations.
    3. From last operation to Cmax.
    """

    operations = get_machine_operations(

        schedule,

        machine_id

    )


    cmax = schedule.makespan



    # --------------------------------------------------------
    # No operation on machine
    # --------------------------------------------------------

    if not operations:

        if cmax > 0:

            return [cmax]

        return []



    gaps = []



    # --------------------------------------------------------
    # Initial gap
    # --------------------------------------------------------

    first_gap = (

        operations[0].start

    )


    if first_gap > 0:

        gaps.append(

            first_gap

        )



    # --------------------------------------------------------
    # Internal gaps
    # --------------------------------------------------------

    for index in range(

        len(operations) - 1

    ):


        gap = (

            operations[index + 1].start

            -

            operations[index].finish

        )


        if gap > 0:

            gaps.append(

                gap

            )



    # --------------------------------------------------------
    # Final gap
    # --------------------------------------------------------

    final_gap = (

        cmax

        -

        operations[-1].finish

    )


    if final_gap > 0:

        gaps.append(

            final_gap

        )


    return gaps



# ============================================================
# IDLE Energy
# ============================================================

def calculate_idle_energy(
        gap,
        machine_energy
):

    """
    Idle-state energy:

        E_I =
            P_I,j * gap
    """

    return (

        machine_energy.idle_power

        *

        gap

    )



# ============================================================
# STANDBY Energy
# ============================================================

def calculate_standby_energy(
        gap,
        machine_energy
):

    """
    Standby-state energy:

        E_SB =

            P_rampup,standby,j
            * t_rampup,standby,j

            +

            P_S,j
            * (
                gap
                -
                t_rampup,standby,j
            )

    Standby is feasible only when:

        gap >= t_rampup,standby,j
    """

    ramp_time = (

        machine_energy.ramp_standby_time

    )


    if gap < ramp_time:

        return float("inf")



    standby_time = (

        gap

        -

        ramp_time

    )



    return (

        machine_energy.ramp_standby_power

        *

        ramp_time


        +

        machine_energy.standby_power

        *

        standby_time

    )



# ============================================================
# OFF Energy
# ============================================================

def calculate_off_energy(
        gap,
        machine_energy
):

    """
    Off-state energy:

        E_OFF =

            P_rampup,off,j
            *
            t_rampup,off,j

    OFF state itself is assumed to consume zero energy.

    OFF is feasible only when:

        gap >= t_rampup,off,j
    """

    ramp_time = (

        machine_energy.ramp_off_time

    )


    if gap < ramp_time:

        return float("inf")



    return (

        machine_energy.ramp_off_power

        *

        ramp_time

    )



# ============================================================
# Machine State Decision
# Eq. (1)-(6)
# ============================================================

def choose_idle_state(
        gap,
        machine_energy
):

    """
    Select the minimum-energy feasible state:

        IDLE
        STANDBY
        OFF

    If two states have exactly the same energy,
    IDLE is preferred because it appears first.
    """


    if gap <= 0:

        return (

            "NO_GAP",

            0.0

        )



    idle_energy = calculate_idle_energy(

        gap,

        machine_energy

    )


    standby_energy = calculate_standby_energy(

        gap,

        machine_energy

    )


    off_energy = calculate_off_energy(

        gap,

        machine_energy

    )



    energies = {

        "IDLE":
            idle_energy,

        "STANDBY":
            standby_energy,

        "OFF":
            off_energy

    }



    selected_state = min(

        energies,

        key=energies.get

    )



    return (

        selected_state,

        energies[
            selected_state
        ]

    )



# ============================================================
# NEC
# Wei Eq. (11)
# ============================================================

def calculate_NEC(
        schedule,
        machine_models
):

    """
    Wei Eq. (11):

        NEC =
        sum_j [

            P_I,j * t_I,j

            +

            P_S,j * t_S,j

            +

            n_rampup,standby,j
            * P_rampup,standby,j
            * t_rampup,standby,j

            +

            n_rampup,off,j
            * P_rampup,off,j
            * t_rampup,off,j

        ]
    """


    total_nec = 0.0



    for machine_id in range(

        len(machine_models)

    ):


        machine = (

            machine_models[
                machine_id
            ]

        )



        gaps = calculate_machine_idle_periods(

            schedule,

            machine_id

        )



        idle_time = 0.0

        standby_time = 0.0

        number_ramp_standby = 0

        number_ramp_off = 0



        for gap in gaps:


            state, _ = choose_idle_state(

                gap,

                machine

            )



            # =================================================
            # IDLE
            # =================================================

            if state == "IDLE":

                idle_time += (

                    gap

                )



            # =================================================
            # STANDBY
            # =================================================

            elif state == "STANDBY":

                current_standby_time = (

                    gap

                    -

                    machine.ramp_standby_time

                )


                # This should always be non-negative because
                # choose_idle_state() checks feasibility.

                if current_standby_time < 0:

                    raise RuntimeError(

                        "Negative standby duration detected."

                    )


                standby_time += (

                    current_standby_time

                )


                number_ramp_standby += 1



            # =================================================
            # OFF
            # =================================================

            elif state == "OFF":

                number_ramp_off += 1



        # ====================================================
        # Wei Eq. (11)
        # ====================================================

        machine_nec = (

            # P_I,j * t_I,j

            machine.idle_power

            *

            idle_time


            +

            # P_S,j * t_S,j

            machine.standby_power

            *

            standby_time


            +

            # standby ramp-up term

            number_ramp_standby

            *

            machine.ramp_standby_power

            *

            machine.ramp_standby_time


            +

            # off ramp-up term

            number_ramp_off

            *

            machine.ramp_off_power

            *

            machine.ramp_off_time

        )



        total_nec += (

            machine_nec

        )



    return total_nec



# ============================================================
# Processing Energy
# ============================================================

def calculate_processing_energy(
        operation,
        machine_energy
):

    """
    Processing-energy component:

        E_W =
            P_W,j
            *
            processing time
    """


    processing_time = (

        operation.finish

        -

        operation.start

    )



    return (

        machine_energy.processing_power

        *

        processing_time

    )



# ============================================================
# Eq. (7)
# Total Energy Without Energy-Saving Strategy
# ============================================================

def calculate_total_energy_without_energy_saving(
        schedule,
        machine_models
):

    """
    Wei Eq. (7):

        processing energy

        +

        ordinary idle energy
    """


    total_processing_energy = 0.0



    for operation in schedule.operations:


        machine = (

            machine_models[
                operation.machine
            ]

        )


        total_processing_energy += (

            calculate_processing_energy(

                operation,

                machine

            )

        )



    total_idle_energy = 0.0


    cmax = schedule.makespan



    for machine_id in range(

        len(machine_models)

    ):


        machine_processing_time = 0.0



        for operation in schedule.operations:


            if operation.machine == machine_id:


                machine_processing_time += (

                    operation.finish

                    -

                    operation.start

                )



        non_processing_time = (

            cmax

            -

            machine_processing_time

        )


        if non_processing_time < 0:

            raise RuntimeError(

                "Negative non-processing time detected."

            )



        total_idle_energy += (

            non_processing_time

            *

            machine_models[
                machine_id
            ].idle_power

        )



    return (

        total_processing_energy

        +

        total_idle_energy

    )



# ============================================================
# NEC Without Energy-Saving Strategy
# ============================================================

def calculate_NEC_without_energy_saving(
        schedule,
        machine_models
):

    """
    Non-processing component of Wei Eq. (7).

    All non-processing time is treated as IDLE.
    """


    cmax = schedule.makespan


    total_nec = 0.0



    for machine_id in range(

        len(machine_models)

    ):


        processing_time = 0.0



        for operation in schedule.operations:


            if operation.machine == machine_id:


                processing_time += (

                    operation.finish

                    -

                    operation.start

                )



        idle_time = (

            cmax

            -

            processing_time

        )


        if idle_time < 0:

            raise RuntimeError(

                "Negative idle time detected."

            )



        total_nec += (

            idle_time

            *

            machine_models[
                machine_id
            ].idle_power

        )



    return total_nec



# ============================================================
# Eq. (8)
# Total Energy After Energy-Saving Strategy
# ============================================================

def calculate_total_energy(
        schedule,
        machine_models
):

    """
    Wei Eq. (8):

        total energy
        =
        processing energy
        +
        NEC

    The optimization objective is NEC.
    """


    processing_energy = 0.0



    for operation in schedule.operations:


        machine = (

            machine_models[
                operation.machine
            ]

        )


        processing_energy += (

            calculate_processing_energy(

                operation,

                machine

            )

        )



    nec = calculate_NEC(

        schedule,

        machine_models

    )



    return (

        processing_energy

        +

        nec

    )