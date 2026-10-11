"""
Wei2021 Reproduction

Population Evaluation

Strict objectives:

1. Makespan
2. TWET
3. NEC

No diagnostic TWET variants are used here.
"""


from decoder import decode, decode_right_shift

from objectives import (
    calculate_makespan,
    calculate_twet
)

from energy import calculate_NEC



# ============================================================
# Evaluate One Individual
# ============================================================

def evaluate_individual(
        instance,
        chromosome,
        due_dates,
        weights,
        machine_models,
        right_shift=False
):

    """
    Decode one chromosome and calculate the
    three Wei objectives:

        Cmax
        TWET
        NEC

    `right_shift` selects the scoring decoder. It defaults to False, using the
    validated semi-active `decode` and reproducing the baseline byte-for-byte.

    When True, the energy-aware right-shift refinement (MEMETIC_DESIGN.md
    §3b/§4.3a) is also computed. Because the right-shift transform preserves
    every job's completion time, Cmax and TWET are identical either way — only
    NEC can change. The right-shifted NEC is kept only when it does NOT exceed
    the semi-active NEC, so the refinement is Pareto-safe by construction (a
    pure NEC decrease or no-op, never a regression). This flag MUST be set the
    same for every method compared, or the comparison is unfair.
    """


    # --------------------------------------------------------
    # Semi-active (validated) decode: fixes Cmax, TWET and the
    # baseline NEC.
    # --------------------------------------------------------

    schedule = decode(

        instance,

        chromosome

    )



    # ========================================================
    # Objective 1: Makespan
    # Wei Eq. (9)
    # ========================================================

    cmax = calculate_makespan(

        schedule

    )



    # ========================================================
    # Objective 2: TWET
    # Wei Eq. (10), (16), (17)
    # ========================================================

    twet = calculate_twet(

        schedule,

        due_dates,

        weights,

        # Use each job's OWN completion time C_i (Wei Eq. 16/17
        # corrected). The default "makespan" reading charges every
        # job at the global Cmax, which makes TWET a perfect linear
        # function of Cmax (corr = 1.000 across all instances) and
        # therefore not an independent objective. per_job restores
        # TWET as a genuine third objective.
        completion_mode="per_job"

    )



    # ========================================================
    # Objective 3: NEC
    # Wei Eq. (11)
    # ========================================================

    nec = calculate_NEC(

        schedule,

        machine_models

    )


    # --------------------------------------------------------
    # Energy-aware right-shift refinement (Pareto-safe).
    #
    # Right-shift preserves every C_i, so Cmax/TWET are unchanged;
    # it only redistributes idle. Keep the refined NEC only when it
    # does not exceed the semi-active NEC (see decode_right_shift's
    # docstring: the raw transform can fragment idle as well as merge
    # it, so the min() here is what makes it a pure decrease-or-no-op).
    # base_schedule is reused, so this costs no extra decode.
    # --------------------------------------------------------

    if right_shift:

        shifted = decode_right_shift(
            instance, chromosome, base_schedule=schedule
        )

        nec_shifted = calculate_NEC(shifted, machine_models)

        if nec_shifted < nec:
            nec = nec_shifted



    return {

        "chromosome":
            chromosome,

        "Cmax":
            cmax,

        "TWET":
            twet,

        "NEC":
            nec

    }



# ============================================================
# Evaluate Population
# ============================================================

def evaluate_population(
        instance,
        population,
        due_dates,
        weights,
        machine_models,
        right_shift=False
):

    results = []


    for index, individual in enumerate(

        population

    ):


        result = evaluate_individual(

            instance,

            individual,

            due_dates,

            weights,

            machine_models,

            right_shift=right_shift

        )


        # ----------------------------------------------------
        # Internal ID for NSGA-III handling
        # ----------------------------------------------------

        result["id"] = index


        results.append(

            result

        )


    return results