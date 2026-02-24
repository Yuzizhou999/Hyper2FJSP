from enum import Enum


class ObjectiveFn(Enum):
    MAKESPAN = "makespan"
    TOTAL_TARDINESS = "total_tardiness"
    TOTAL_EARLINESS = "total_earliness"
    AVERAGE_FLOWTIME = "average_flowtime"
    NEGATIVE_MAKESPAN = "negative_makespan"
    COSTS = "costs"


class ModelArchitecture(Enum):
    DANIEL = "daniel"  # Default single-objective model
    MO_DANIEL_ENC_WEIGHT_INPUT = "mo_daniel_enc_weight_input"
    MO_DANIEL_CONDITIONAL_OP_AND_MCH = "mo_daniel_conditional_op_and_mch"
    MO_DANIEL_CONDITIONAL_OP_AND_MCH_FEA_INPUT = (
        "mo_daniel_conditional_op_and_mch_fea_input"
    )
    # Hypernetwork-based model
    HYPER_DANIEL = "hyper_daniel"

    @classmethod
    def from_string(cls, s: str):
        for arch in cls:
            if arch.value == s.lower():
                return arch
        raise ValueError(f"Invalid model architecture: {s}")
