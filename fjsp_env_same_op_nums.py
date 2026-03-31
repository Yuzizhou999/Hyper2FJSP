import copy
import sys
from dataclasses import dataclass

import numpy as np
import numpy.ma as ma
import torch

from enums import ObjectiveFn
from params import configs


@dataclass
class EnvState:
    """
    state definition
    """

    fea_j_tensor = torch.empty(0)
    op_mask_tensor = torch.empty(0)
    fea_m_tensor = torch.empty(0)
    mch_mask_tensor = torch.empty(0)
    dynamic_pair_mask_tensor = torch.empty(0)
    comp_idx_tensor = torch.empty(0)
    candidate_tensor = torch.empty(0)
    fea_pairs_tensor = torch.empty(0)
    event_context_tensor = torch.empty(0)
    event_mch_tensor = torch.empty(0)

    device: torch.device = torch.device(configs.device)

    def update(
        self,
        fea_j: np.ndarray,
        op_mask: np.ndarray,
        fea_m: np.ndarray,
        mch_mask: np.ndarray,
        dynamic_pair_mask: np.ndarray,
        comp_idx: np.ndarray,
        candidate: np.ndarray,
        fea_pairs: np.ndarray,
        event_context: np.ndarray | None = None,
        event_mch: np.ndarray | None = None,
    ) -> None:
        """
            update the state information
        :param fea_j: input operation feature vectors with shape [sz_b, N, 10]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 8]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :return:
        """
        device = self.device
        self.fea_j_tensor = torch.from_numpy(np.copy(fea_j)).float().to(device)
        self.fea_m_tensor = torch.from_numpy(np.copy(fea_m)).float().to(device)
        self.fea_pairs_tensor = torch.from_numpy(np.copy(fea_pairs)).float().to(device)

        self.op_mask_tensor = torch.from_numpy(np.copy(op_mask)).to(device)
        self.candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        self.mch_mask_tensor = torch.from_numpy(np.copy(mch_mask)).float().to(device)
        self.comp_idx_tensor = torch.from_numpy(np.copy(comp_idx)).to(device)
        self.dynamic_pair_mask_tensor = torch.from_numpy(np.copy(dynamic_pair_mask)).to(
            device
        )
        if event_context is None:
            self.event_context_tensor = torch.empty((fea_j.shape[0], 0)).float().to(
                device
            )
        else:
            self.event_context_tensor = (
                torch.from_numpy(np.copy(event_context)).float().to(device)
            )
        if event_mch is None:
            self.event_mch_tensor = torch.empty((fea_j.shape[0], 0, 0)).float().to(
                device
            )
        else:
            self.event_mch_tensor = torch.from_numpy(np.copy(event_mch)).float().to(
                device
            )

    def print_shape(self) -> None:
        print(self.fea_j_tensor.shape)
        print(self.op_mask_tensor.shape)
        print(self.candidate_tensor.shape)
        print(self.fea_m_tensor.shape)
        print(self.mch_mask_tensor.shape)
        print(self.comp_idx_tensor.shape)
        print(self.dynamic_pair_mask_tensor.shape)
        print(self.fea_pairs_tensor.shape)


class FJSPEnvForSameOpNums:
    """
    a batch of fjsp environments that have the same number of operations

    let E/N/J/M denote the number of envs/operations/jobs/machines
    Remark: The index of operations has been rearranged in natural order
    eg. {O_{11}, O_{12}, O_{13}, O_{21}, O_{22}}  <--> {0,1,2,3,4}

    Attributes:

    job_length: the number of operations in each job (shape [J])
    op_pt: the processing time matrix with shape [N, M],
            where op_pt[i,j] is the processing time of the ith operation
            on the jth machine or 0 if $O_i$ can not process on $M_j$

    candidate: the index of candidates  [sz_b, J]
    fea_j: input operation feature vectors with shape [sz_b, N, 8]
    op_mask: used for masking nonexistent predecessors/successor
                    (with shape [sz_b, N, 3])
    fea_m: input operation feature vectors with shape [sz_b, M, 6]
    mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
    comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                the value of comp_idx[i, k, q, j] (any i) means whether
                machine $M_k$ and $M_q$ are competing for candidate[i,j]
    dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking incompatible op-mch pairs
    fea_pairs: pair features with shape [sz_b, J, M, 8]
    """

    def __init__(
        self, n_j, n_m, objective_fn: ObjectiveFn = ObjectiveFn.AVERAGE_FLOWTIME
    ):
        """
        :param n_j: the number of jobs
        :param n_m: the number of machines
        """
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.old_state = EnvState()

        # the dimension of operation raw features
        if objective_fn in [ObjectiveFn.TOTAL_TARDINESS, ObjectiveFn.AVERAGE_FLOWTIME]:
            self.op_fea_dim = 11
        else:
            self.op_fea_dim = 10
        # the dimension of machine raw features
        self.mch_fea_dim = 8
        # The objective function to be optimized
        self.objective = objective_fn

    def set_static_properties(self):
        """
        define static properties
        """
        self.multi_env_mch_diag = np.tile(
            np.expand_dims(np.eye(self.number_of_machines, dtype=bool), axis=0),
            (self.number_of_envs, 1, 1),
        )

        self.env_idxs = np.arange(self.number_of_envs)
        self.env_job_idx = self.env_idxs.repeat(self.number_of_jobs).reshape(
            self.number_of_envs, self.number_of_jobs
        )
        self.op_idx = np.arange(self.number_of_ops)[np.newaxis, :]

    def set_initial_data(self, job_length_list, op_pt_list, deadline_alpha=1.0):
        """
            initialize the data of the instances

        :param job_length_list: the list of 'job_length'
        :param op_pt_list: the list of 'op_pt'
        """

        self.number_of_envs = len(job_length_list)
        self.job_length = np.array(job_length_list)
        self.op_pt = np.array(op_pt_list)
        self.number_of_ops = self.op_pt.shape[1]
        self.number_of_machines = op_pt_list[0].shape[1]
        self.number_of_jobs = job_length_list[0].shape[0]

        self.set_static_properties()

        # [E, N, M]
        self.pt_lower_bound = np.min(self.op_pt)
        self.pt_upper_bound = np.max(self.op_pt)
        self.true_op_pt = np.copy(self.op_pt)

        self.costs = (
            np.tile(
                np.max(self.true_op_pt, axis=(1, 2))[:, np.newaxis, np.newaxis],
                (1, self.true_op_pt.shape[1], self.true_op_pt.shape[2]),
            )
            - self.true_op_pt
        )
        self.true_costs = np.copy(self.costs)
        # normalize the costs so reward scaling stays comparable across instances
        self.costs = (self.costs - np.min(self.costs)) / (
            np.max(self.costs) - np.min(self.costs) + 1e-8
        )

        # normalize the processing time
        self.op_pt = (self.op_pt - self.pt_lower_bound) / (
            self.pt_upper_bound - self.pt_lower_bound + 1e-8
        )

        # bool 3-d array formulating the compatible relation with shape [E,N,M]
        self.process_relation = self.op_pt != 0
        self.reverse_process_relation = ~self.process_relation

        # number of compatible machines of each operation ([E,N])
        self.compatible_op = np.sum(self.process_relation, 2)
        # number of operations that each machine can process ([E,M])
        self.compatible_mch = np.sum(self.process_relation, 1)

        self.unmasked_op_pt = np.copy(self.op_pt)

        head_op_id = np.zeros((self.number_of_envs, 1))

        # the index of first operation of each job ([E,J])
        self.job_first_op_id = np.concatenate(
            [head_op_id, np.cumsum(self.job_length, axis=1)[:, :-1]], axis=1
        ).astype("int")
        # the index of last operation of each job ([E,J])
        self.job_last_op_id = self.job_first_op_id + self.job_length - 1

        self.true_job_start_times = np.full_like(
            self.job_first_op_id, np.inf, dtype=np.float64
        )
        self.job_start_times = np.copy(self.true_job_start_times)

        self.initial_vars()

        self.init_op_mask()

        self.op_pt = ma.array(self.op_pt, mask=self.reverse_process_relation)

        self.costs = ma.array(self.costs, mask=self.reverse_process_relation)
        self.true_costs = ma.array(self.true_costs, mask=self.reverse_process_relation)

        self.cost_lb = np.sum(np.min(self.costs, axis=-1), axis=-1).data
        self.true_cost_lb = np.sum(np.min(self.true_costs, axis=-1), axis=-1)

        """
            compute operation raw features
        """
        self.op_mean_pt = np.mean(self.op_pt, axis=2).data

        self.op_min_pt = np.min(self.op_pt, axis=-1).data
        self.true_op_min_pt = np.min(
            ma.array(self.true_op_pt, mask=self.reverse_process_relation), axis=-1
        ).data
        self.op_max_pt = np.max(self.op_pt, axis=-1).data
        self.pt_span = self.op_max_pt - self.op_min_pt
        # [E, M]
        self.mch_min_pt = np.max(self.op_pt, axis=1).data
        self.mch_max_pt = np.max(self.op_pt, axis=1)

        # the estimated lower bound of complete time of operations
        self.op_ct_lb = copy.deepcopy(self.op_min_pt)
        self.true_op_ct_lb = copy.deepcopy(self.true_op_min_pt)
        for k in range(self.number_of_envs):
            for i in range(self.number_of_jobs):
                self.op_ct_lb[k][
                    self.job_first_op_id[k][i] : self.job_last_op_id[k][i] + 1
                ] = np.cumsum(
                    self.op_ct_lb[k][
                        self.job_first_op_id[k][i] : self.job_last_op_id[k][i] + 1
                    ]
                )
                self.true_op_ct_lb[k][
                    self.job_first_op_id[k][i] : self.job_last_op_id[k][i] + 1
                ] = np.cumsum(
                    self.true_op_ct_lb[k][
                        self.job_first_op_id[k][i] : self.job_last_op_id[k][i] + 1
                    ]
                )

        if deadline_alpha is not None:
            self.generate_deadlines(deadline_alpha)

        # job remaining number of operations
        self.op_match_job_left_op_nums = np.array(
            [
                np.repeat(self.job_length[k], repeats=self.job_length[k])
                for k in range(self.number_of_envs)
            ]
        )
        self.job_remain_work = []
        for k in range(self.number_of_envs):
            self.job_remain_work.append(
                [
                    np.sum(
                        self.op_mean_pt[k][
                            self.job_first_op_id[k][i] : self.job_last_op_id[k][i] + 1
                        ]
                    )
                    for i in range(self.number_of_jobs)
                ]
            )

        self.op_match_job_remain_work = np.array(
            [
                np.repeat(self.job_remain_work[k], repeats=self.job_length[k])
                for k in range(self.number_of_envs)
            ]
        )
        # If the objective function is total tardiness, we want to keep track of
        # the lower bound of the lateness of the job and give it as a feature for
        # the operations
        self.compute_tardiness_features()
        # If the objective is the average flowtime, we want to keep track of the
        # lower bound of the lateness of the job and give it as a feature for the
        # operations
        self.compute_flowtime_features()
        self.construct_op_features()

        # shape reward
        self.set_initial_reward_quality()

        self.previous_obj_estimate = np.copy(self.init_quality)
        """
            compute machine raw features
        """
        self.mch_available_op_nums = np.copy(self.compatible_mch)
        self.mch_current_available_op_nums = np.copy(self.compatible_mch)
        # [E, J, M]
        self.candidate_pt = np.array(
            [
                self.unmasked_op_pt[k][self.candidate[k]]
                for k in range(self.number_of_envs)
            ]
        )

        # construct dynamic pair mask : [E, J, M]
        self.dynamic_pair_mask = self.candidate_pt == 0
        self.candidate_process_relation = np.copy(self.dynamic_pair_mask)
        self.mch_current_available_jc_nums = np.sum(
            ~self.candidate_process_relation, axis=1
        )

        self.mch_mean_pt = np.mean(self.op_pt, axis=1).filled(0)
        # construct machine features [E, M, 6]

        # construct 'come_idx' : [E, M, M, J]
        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)
        self.init_mch_mask()
        self.construct_mch_features()

        self.construct_pair_features()

        self.old_state.update(
            self.fea_j,
            self.op_mask,
            self.fea_m,
            self.mch_mask,
            self.dynamic_pair_mask,
            self.comp_idx,
            self.candidate,
            self.fea_pairs,
        )

        # old record
        self.old_op_mask = np.copy(self.op_mask)
        self.old_mch_mask = np.copy(self.mch_mask)
        self.old_op_ct_lb = np.copy(self.op_ct_lb)
        self.old_op_match_job_left_op_nums = np.copy(self.op_match_job_left_op_nums)
        self.old_op_match_job_remain_work = np.copy(self.op_match_job_remain_work)
        self.old_init_quality = np.copy(self.init_quality)
        self.old_candidate_pt = np.copy(self.candidate_pt)
        self.old_candidate_process_relation = np.copy(self.candidate_process_relation)
        self.old_mch_current_available_op_nums = np.copy(
            self.mch_current_available_op_nums
        )
        self.old_mch_current_available_jc_nums = np.copy(
            self.mch_current_available_jc_nums
        )
        self.old_cost_lb = np.copy(self.cost_lb)
        self.old_true_cost_lb = np.copy(self.true_cost_lb)
        # state
        self.state = copy.deepcopy(self.old_state)
        return self.state

    def reset(self):
        """
           reset the environments
        :return: the state
        """
        self.initial_vars()

        # copy the old data
        self.op_mask = np.copy(self.old_op_mask)
        self.mch_mask = np.copy(self.old_mch_mask)
        self.op_ct_lb = np.copy(self.old_op_ct_lb)
        self.op_match_job_left_op_nums = np.copy(self.old_op_match_job_left_op_nums)
        self.op_match_job_remain_work = np.copy(self.old_op_match_job_remain_work)
        self.init_quality = np.copy(self.old_init_quality)
        self.previous_obj_estimate = self.init_quality
        self.candidate_pt = np.copy(self.old_candidate_pt)
        self.candidate_process_relation = np.copy(self.old_candidate_process_relation)
        self.mch_current_available_op_nums = np.copy(
            self.old_mch_current_available_op_nums
        )
        self.mch_current_available_jc_nums = np.copy(
            self.old_mch_current_available_jc_nums
        )
        self.cost_lb = np.copy(self.old_cost_lb)
        self.true_cost_lb = np.copy(self.old_true_cost_lb)
        # copy the old state
        self.state = copy.deepcopy(self.old_state)
        return self.state

    def initial_vars(self):
        """
        initialize variables for further use
        """
        self.step_count = 0
        # the array that records the makespan of all environments
        self.current_makespan = np.full(self.number_of_envs, float("-inf"))
        # the complete time of operations ([E,N])
        self.op_ct = np.zeros((self.number_of_envs, self.number_of_ops))
        self.mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_remain_work = np.zeros((self.number_of_envs, self.number_of_machines))

        self.mch_waiting_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_working_flag = np.zeros((self.number_of_envs, self.number_of_machines))

        self.next_schedule_time = np.zeros(self.number_of_envs)
        self.candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs))

        self.true_op_ct = np.zeros((self.number_of_envs, self.number_of_ops))
        self.true_candidate_free_time = np.zeros(
            (self.number_of_envs, self.number_of_jobs)
        )
        self.true_mch_free_time = np.zeros(
            (self.number_of_envs, self.number_of_machines)
        )

        # Track accumulated actual cost of scheduled operations per environment
        self.actual_cost_sum = np.zeros(self.number_of_envs, dtype=np.float64)

        self.candidate = np.copy(self.job_first_op_id)

        # mask[i,j] : whether the jth job of ith env is scheduled (have no unscheduled operations)
        self.mask = np.full(
            shape=(self.number_of_envs, self.number_of_jobs), fill_value=0, dtype=bool
        )

        self.op_scheduled_flag = np.zeros((self.number_of_envs, self.number_of_ops))
        self.op_waiting_time = np.zeros((self.number_of_envs, self.number_of_ops))
        self.op_remain_work = np.zeros((self.number_of_envs, self.number_of_ops))

        self.op_available_mch_nums = (
            np.copy(self.compatible_op) / self.number_of_machines
        )
        self.pair_free_time = np.zeros(
            (self.number_of_envs, self.number_of_jobs, self.number_of_machines)
        )
        self.remain_process_relation = np.copy(self.process_relation)

        self.delete_mask_fea_j = np.full(
            shape=(self.number_of_envs, self.number_of_ops, self.op_fea_dim),
            fill_value=0,
            dtype=bool,
        )
        # mask[i,j] : whether the jth op of ith env is deleted (from the set $O_u$)
        self.deleted_op_nodes = np.full(
            shape=(self.number_of_envs, self.number_of_ops), fill_value=0, dtype=bool
        )

    def step(self, actions):
        """
            perform the state transition & return the next state and reward
        :param actions: the action list with shape [E]
        :return: the next state, reward and the done flag
        """
        chosen_job = actions // self.number_of_machines
        chosen_mch = actions % self.number_of_machines
        chosen_op = self.candidate[self.env_idxs, chosen_job]

        if (self.reverse_process_relation[self.env_idxs, chosen_op, chosen_mch]).any():
            print(
                f"FJSP_Env.py Error from choosing action: Op {chosen_op} can't be processed by Mch {chosen_mch}"
            )
            sys.exit()

        self.step_count += 1

        # update candidate
        candidate_add_flag = chosen_op != self.job_last_op_id[self.env_idxs, chosen_job]
        self.candidate[self.env_idxs, chosen_job] += candidate_add_flag
        self.mask[self.env_idxs, chosen_job] = 1 - candidate_add_flag

        # the start processing time of chosen operations
        chosen_op_st = np.maximum(
            self.candidate_free_time[self.env_idxs, chosen_job],
            self.mch_free_time[self.env_idxs, chosen_mch],
        )

        self.op_ct[self.env_idxs, chosen_op] = (
            chosen_op_st + self.op_pt[self.env_idxs, chosen_op, chosen_mch]
        )
        self.candidate_free_time[self.env_idxs, chosen_job] = self.op_ct[
            self.env_idxs, chosen_op
        ]
        self.mch_free_time[self.env_idxs, chosen_mch] = self.op_ct[
            self.env_idxs, chosen_op
        ]

        true_chosen_op_st = np.maximum(
            self.true_candidate_free_time[self.env_idxs, chosen_job],
            self.true_mch_free_time[self.env_idxs, chosen_mch],
        )

        # Update the job start times
        # Check if operation is the first operation of the job
        is_first_operation = np.isin(chosen_op, self.job_first_op_id)
        self.true_job_start_times[
            is_first_operation, chosen_job[is_first_operation]
        ] = true_chosen_op_st[is_first_operation]
        self.job_start_times[is_first_operation, chosen_job[is_first_operation]] = (
            chosen_op_st[is_first_operation]
        )

        self.true_op_ct[self.env_idxs, chosen_op] = (
            true_chosen_op_st + self.true_op_pt[self.env_idxs, chosen_op, chosen_mch]
        )
        self.true_candidate_free_time[self.env_idxs, chosen_job] = self.true_op_ct[
            self.env_idxs, chosen_op
        ]
        self.true_mch_free_time[self.env_idxs, chosen_mch] = self.true_op_ct[
            self.env_idxs, chosen_op
        ]

        self.current_makespan = np.maximum(
            self.current_makespan, self.true_op_ct[self.env_idxs, chosen_op]
        )

        # update the cost lb
        self.cost_lb += (
            self.costs[self.env_idxs, chosen_op, chosen_mch].data
            - np.min(self.costs[self.env_idxs, chosen_op], axis=-1).data
        )
        self.true_cost_lb += self.true_costs[
            self.env_idxs, chosen_op, chosen_mch
        ] - np.min(self.true_costs[self.env_idxs, chosen_op], axis=-1)

        # Accumulate actual costs of scheduled operations for simple reward strategy
        self.actual_cost_sum += np.ma.filled(
            self.true_costs[self.env_idxs, chosen_op, chosen_mch], 0
        )

        # update the candidate message
        mask_temp = candidate_add_flag
        self.candidate_pt[mask_temp, chosen_job[mask_temp]] = self.unmasked_op_pt[
            mask_temp, chosen_op[mask_temp] + 1
        ]
        self.candidate_process_relation[mask_temp, chosen_job[mask_temp]] = (
            self.reverse_process_relation[mask_temp, chosen_op[mask_temp] + 1]
        )
        self.candidate_process_relation[~mask_temp, chosen_job[~mask_temp]] = 1

        # compute the next schedule time

        # [E, J, M]
        candidateFT_for_compare = np.expand_dims(self.candidate_free_time, axis=2)
        mchFT_for_compare = np.expand_dims(self.mch_free_time, axis=1)
        self.pair_free_time = np.maximum(candidateFT_for_compare, mchFT_for_compare)

        schedule_matrix = ma.array(
            self.pair_free_time, mask=self.candidate_process_relation
        )

        self.next_schedule_time = np.min(
            schedule_matrix.reshape(self.number_of_envs, -1), axis=1
        ).data

        self.remain_process_relation[self.env_idxs, chosen_op] = 0
        self.op_scheduled_flag[self.env_idxs, chosen_op] = 1

        """
            update the mask for deleting nodes
        """
        self.deleted_op_nodes = np.logical_and(
            (self.op_ct <= self.next_schedule_time[:, np.newaxis]),
            self.op_scheduled_flag,
        )
        self.delete_mask_fea_j = np.tile(
            self.deleted_op_nodes[:, :, np.newaxis], (1, 1, self.op_fea_dim)
        )

        """
            update the state
        """
        self.update_op_mask()

        # update operation raw features
        diff = (
            self.op_ct[self.env_idxs, chosen_op]
            - self.op_ct_lb[self.env_idxs, chosen_op]
        )

        mask1 = (self.op_idx >= chosen_op[:, np.newaxis]) & (
            self.op_idx
            < (self.job_last_op_id[self.env_idxs, chosen_job] + 1)[:, np.newaxis]
        )
        self.op_ct_lb[mask1] += np.tile(diff[:, np.newaxis], (1, self.number_of_ops))[
            mask1
        ]

        mask2 = (
            self.op_idx
            >= (self.job_first_op_id[self.env_idxs, chosen_job])[:, np.newaxis]
        ) & (
            self.op_idx
            < (self.job_last_op_id[self.env_idxs, chosen_job] + 1)[:, np.newaxis]
        )
        self.op_match_job_left_op_nums[mask2] -= 1
        self.op_match_job_remain_work[mask2] -= np.tile(
            self.op_mean_pt[self.env_idxs, chosen_op][:, np.newaxis],
            (1, self.number_of_ops),
        )[mask2]

        self.op_waiting_time = np.zeros((self.number_of_envs, self.number_of_ops))
        self.op_waiting_time[self.env_job_idx, self.candidate] = (
            1 - self.mask
        ) * np.maximum(
            np.expand_dims(self.next_schedule_time, axis=1) - self.candidate_free_time,
            0,
        ) + self.mask * self.op_waiting_time[self.env_job_idx, self.candidate]

        self.op_remain_work = np.maximum(
            self.op_ct - np.expand_dims(self.next_schedule_time, axis=1), 0
        )

        # If the objective function is total tardiness, we want to keep track of
        # the lower bound of the lateness of the job and give it as a feature for
        # the operations
        self.compute_tardiness_features()
        # If the objective is the average flowtime, we want to keep track of the
        # lower bound of the lateness of the job and give it as a feature for the
        # operations
        self.compute_flowtime_features()

        self.construct_op_features()

        # update dynamic pair mask
        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)

        self.unavailable_pairs = (
            self.pair_free_time > self.next_schedule_time[:, np.newaxis, np.newaxis]
        )

        self.dynamic_pair_mask = np.logical_or(
            self.dynamic_pair_mask, self.unavailable_pairs
        )

        # update comp_idx
        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)

        self.update_mch_mask()

        # update machine raw features
        self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
        self.mch_current_available_op_nums -= self.process_relation[
            self.env_idxs, chosen_op
        ]

        mch_free_duration = (
            np.expand_dims(self.next_schedule_time, axis=1) - self.mch_free_time
        )
        mch_free_flag = mch_free_duration < 0
        self.mch_working_flag = mch_free_flag + 0
        self.mch_waiting_time = (1 - mch_free_flag) * mch_free_duration

        self.mch_remain_work = np.maximum(-mch_free_duration, 0)

        self.construct_mch_features()

        self.construct_pair_features()

        # compute the reward : R_t = C_{LB}(s_{t}) - C_{LB}(s_{t+1})
        reward = self.compute_reward()

        # update the state
        self.state.update(
            self.fea_j,
            self.op_mask,
            self.fea_m,
            self.mch_mask,
            self.dynamic_pair_mask,
            self.comp_idx,
            self.candidate,
            self.fea_pairs,
        )

        return self.state, np.array(reward), self.done()

    def done(self):
        """
        compute the done flag
        """
        return np.ones(self.number_of_envs) * (self.step_count >= self.number_of_ops)

    def construct_op_features(self):
        """
        construct operation raw features
        """
        self.fea_j = np.stack(
            (
                self.op_scheduled_flag,
                self.op_ct_lb,
                self.op_min_pt,
                self.pt_span,
                self.op_mean_pt,
                self.op_waiting_time,
                self.op_remain_work,
                self.op_match_job_left_op_nums,
                self.op_match_job_remain_work,
                self.op_available_mch_nums,
            ),
            axis=2,
        )
        if self.objective == ObjectiveFn.TOTAL_TARDINESS:
            self.fea_j = np.concatenate(
                (self.fea_j, self.op_lateness_lb[..., None]), axis=2
            )
        if self.objective == ObjectiveFn.AVERAGE_FLOWTIME:
            self.fea_j = np.concatenate(
                (self.fea_j, self.op_flowtime_lb[..., None]), axis=2
            )
        if self.step_count != self.number_of_ops:
            self.norm_op_features()

    def norm_op_features(self):
        """
        normalize operation raw features (across the second dimension)
        """
        self.fea_j[self.delete_mask_fea_j] = 0
        num_delete_nodes = np.count_nonzero(self.deleted_op_nodes, axis=1)
        num_delete_nodes = num_delete_nodes[:, np.newaxis]
        num_left_nodes = self.number_of_ops - num_delete_nodes
        mean_fea_j = np.sum(self.fea_j, axis=1) / num_left_nodes
        temp = np.where(
            self.delete_mask_fea_j, mean_fea_j[:, np.newaxis, :], self.fea_j
        )
        var_fea_j = np.var(temp, axis=1)

        std_fea_j = np.sqrt(var_fea_j * self.number_of_ops / num_left_nodes)

        self.fea_j = (temp - mean_fea_j[:, np.newaxis, :]) / (
            std_fea_j[:, np.newaxis, :] + 1e-8
        )

    def construct_mch_features(self):
        """
        construct machine raw features
        """
        self.fea_m = np.stack(
            (
                self.mch_current_available_jc_nums,
                self.mch_current_available_op_nums,
                self.mch_min_pt,
                self.mch_mean_pt,
                self.mch_waiting_time,
                self.mch_remain_work,
                self.mch_free_time,
                self.mch_working_flag,
            ),
            axis=2,
        )

        if self.step_count != self.number_of_ops:
            self.norm_machine_features()

    def norm_machine_features(self):
        """
        normalize machine raw features (across the second dimension)
        """
        self.fea_m[self.delete_mask_fea_m] = 0
        num_delete_mchs = np.count_nonzero(self.delete_mask_fea_m[:, :, 0], axis=1)
        num_delete_mchs = num_delete_mchs[:, np.newaxis]
        num_left_mchs = self.number_of_machines - num_delete_mchs
        mean_fea_m = np.sum(self.fea_m, axis=1) / num_left_mchs
        temp = np.where(
            self.delete_mask_fea_m, mean_fea_m[:, np.newaxis, :], self.fea_m
        )
        var_fea_m = np.var(temp, axis=1)
        std_fea_m = np.sqrt(var_fea_m * self.number_of_machines / num_left_mchs)

        self.fea_m = (temp - mean_fea_m[:, np.newaxis, :]) / (
            std_fea_m[:, np.newaxis, :] + 1e-8
        )

    def construct_pair_features(self):
        """
        construct pair features
        """
        remain_op_pt = ma.array(self.op_pt, mask=~self.remain_process_relation)

        chosen_op_max_pt = np.expand_dims(
            self.op_max_pt[self.env_job_idx, self.candidate], axis=-1
        )

        max_remain_op_pt = np.max(
            np.max(remain_op_pt, axis=1, keepdims=True), axis=2, keepdims=True
        ).filled(0 + 1e-8)

        mch_max_remain_op_pt = np.max(remain_op_pt, axis=1, keepdims=True).filled(
            0 + 1e-8
        )

        pair_max_pt = (
            np.max(
                np.max(self.candidate_pt, axis=1, keepdims=True), axis=2, keepdims=True
            )
            + 1e-8
        )

        mch_max_candidate_pt = np.max(self.candidate_pt, axis=1, keepdims=True) + 1e-8

        pair_wait_time = (
            self.op_waiting_time[self.env_job_idx, self.candidate][:, :, np.newaxis]
            + self.mch_waiting_time[:, np.newaxis, :]
        )

        chosen_job_remain_work = (
            np.expand_dims(
                self.op_match_job_remain_work[self.env_job_idx, self.candidate], axis=-1
            )
            + 1e-8
        )

        self.fea_pairs = np.stack(
            (
                self.candidate_pt,
                self.candidate_pt / chosen_op_max_pt,
                self.candidate_pt / mch_max_candidate_pt,
                self.candidate_pt / max_remain_op_pt,
                self.candidate_pt / mch_max_remain_op_pt,
                self.candidate_pt / pair_max_pt,
                self.candidate_pt / chosen_job_remain_work,
                pair_wait_time,
            ),
            axis=-1,
        )

    def update_mch_mask(self):
        """
        update 'mch_mask'
        """
        self.mch_mask = (
            self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        )
        self.delete_mask_fea_m = np.tile(
            ~(np.sum(self.mch_mask, keepdims=True, axis=-1).astype(bool)),
            (1, 1, self.mch_fea_dim),
        )
        self.mch_mask[self.multi_env_mch_diag] = 1

    def init_mch_mask(self):
        """
        initialize 'mch_mask'
        """
        self.mch_mask = (
            self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        )
        self.delete_mask_fea_m = np.tile(
            ~(np.sum(self.mch_mask, keepdims=True, axis=-1).astype(bool)),
            (1, 1, self.mch_fea_dim),
        )
        self.mch_mask[self.multi_env_mch_diag] = 1

    def init_op_mask(self):
        """
        initialize 'op_mask'
        """
        self.op_mask = np.full(
            shape=(self.number_of_envs, self.number_of_ops, 3),
            fill_value=0,
            dtype=np.float32,
        )
        self.op_mask[self.env_job_idx, self.job_first_op_id, 0] = 1
        self.op_mask[self.env_job_idx, self.job_last_op_id, 2] = 1

    def update_op_mask(self):
        """
        update 'op_mask'
        """
        object_mask = np.zeros_like(self.op_mask)
        object_mask[:, :, 2] = self.deleted_op_nodes
        object_mask[:, 1:, 0] = self.deleted_op_nodes[:, :-1]
        self.op_mask = np.logical_or(object_mask, self.op_mask).astype(np.float32)

    def logic_operator(self, x, flagT=True):
        """
            a customized operator for computing some masks
        :param x: a 3-d array with shape [s,a,b]
        :param flagT: whether transpose x in the last two dimensions
        :return:  a 4-d array c, where c[i,j,k,l] = x[i,j,l] & x[i,k,l] for each i,j,k,l
        """
        if flagT:
            x = x.transpose(0, 2, 1)
        d1 = np.expand_dims(x, 2)
        d2 = np.expand_dims(x, 1)

        return np.logical_and(d1, d2).astype(np.float32)

    def compute_flowtimes(self):
        """
        Compute the flowtimes of the jobs. The flowtime of a job is defined as the
        time interval between the start time of the first operation in the job and
        the completion time of the last operation in the job.
        """
        flowtimes = np.zeros((self.number_of_envs, self.number_of_jobs))
        completion_times = self.true_op_ct[
            np.arange(self.number_of_envs)[:, None], self.job_last_op_id
        ]
        flowtimes = completion_times - self.true_job_start_times
        return flowtimes

    def compute_avg_flowtime(self):
        """
        Compute the average flowtime of the jobs.
        """
        flowtimes = self.compute_flowtimes()
        return np.mean(flowtimes, axis=1)

    def compute_max_flowtime(self):
        """
        Compute the maximum flowtime of the jobs.
        """
        flowtimes = self.compute_flowtimes()
        return np.max(flowtimes, axis=1)

    def generate_deadlines(self, alpha=0.5):
        """
        Generate deadlines for a given FJSP instance. The deadline for each job is
        defined as the sum of the minimum processing times of the operations in
        the job multiplied by a factor (1 + alpha), where alpha > 0.
        """
        self.deadlines = self.op_ct_lb[
            np.arange(self.number_of_envs)[:, None], self.job_last_op_id
        ] * (1 + alpha)
        self.true_deadlines = (
            self.deadlines * (self.pt_upper_bound - self.pt_lower_bound + 1e-8)
            + self.pt_lower_bound
        )

    def compute_tardiness(self):
        """
        Compute the tardiness of the jobs. The tardiness of a job is defined as the
        maximum of 0 and the completion time of the job minus its deadline.
        """
        tardiness = np.zeros((self.number_of_envs, self.number_of_jobs))
        completion_times = self.true_op_ct[
            np.arange(self.number_of_envs)[:, None], self.job_last_op_id
        ]
        tardiness = np.maximum(0, completion_times - self.true_deadlines)
        return tardiness

    def compute_avg_tardiness(self):
        """
        Compute the average tardiness of the jobs.
        """
        tardiness = self.compute_tardiness()
        return np.mean(tardiness, axis=1)

    def compute_max_tardiness(self):
        """
        Compute the maximum tardiness of the jobs.
        """
        tardiness = self.compute_tardiness()
        return np.max(tardiness, axis=1)

    def compute_nr_tardy_jobs(self):
        """
        Compute the number of tardy jobs.
        """
        tardiness = self.compute_tardiness()
        return np.sum(tardiness > 0, axis=1)

    def compute_total_tardiness(self):
        """
        Compute the total tardiness of the jobs.
        """
        tardiness = self.compute_tardiness()
        return np.sum(tardiness, axis=1)

    def compute_total_earliness(self):
        """
        Compute the total earliness of the jobs.
        Earliness = max(0, deadline - completion_time)
        """
        earliness = self.compute_earliness()
        return np.sum(earliness, axis=1)

    def compute_earliness(self):
        """
        Helper method to compute earliness per job.
        Returns: [num_envs, num_jobs] array of earliness values
        """
        # Get completion times for the last operation of each job (true scale)
        job_completion_times = self.true_op_ct[
            np.arange(self.number_of_envs)[:, None], self.job_last_op_id
        ]
        # Earliness = deadline - completion (positive part, also in true scale)
        earliness = np.maximum(0, self.true_deadlines - job_completion_times)
        return earliness

    def compute_costs(self):
        """
        Compute the total cost of the jobs.
        """
        return self.true_cost_lb

    def compute_tardiness_features(self):
        if self.objective == ObjectiveFn.TOTAL_TARDINESS:
            # Compute the lateness for each job
            job_lateness = (
                self.op_ct_lb[
                    np.arange(self.number_of_envs)[:, None], self.job_last_op_id
                ]
                - self.deadlines
            )
            # Build an op->job mapping using job_first_op_id and job_last_op_id
            ops = np.arange(self.number_of_ops)[
                None, None, :
            ]  # shape (1, 1, number_of_ops)
            job_first = self.job_first_op_id[
                ..., None
            ]  # shape (number_of_envs, number_of_jobs, 1)
            job_last = self.job_last_op_id[
                ..., None
            ]  # shape (number_of_envs, number_of_jobs, 1)
            job_mask = (ops >= job_first) & (
                ops <= job_last
            )  # shape (number_of_envs, number_of_jobs, number_of_ops)
            op_job_id = job_mask.argmax(axis=1)  # shape (number_of_envs, number_of_ops)
            self.op_lateness_lb = job_lateness[
                np.arange(self.number_of_envs)[:, None], op_job_id
            ]  # shape (number_of_envs, number_of_ops)

    def compute_flowtime_features(self):
        if self.objective == ObjectiveFn.AVERAGE_FLOWTIME:
            job_flowtime_lb = np.where(
                self.job_start_times == np.inf,
                self.op_ct_lb[
                    np.arange(self.number_of_envs)[:, None], self.job_last_op_id
                ],
                self.op_ct_lb[
                    np.arange(self.number_of_envs)[:, None], self.job_last_op_id
                ]
                - self.job_start_times,
            )
            ops = np.arange(self.number_of_ops)[None, None, :]
            job_first = self.job_first_op_id[..., None]
            job_last = self.job_last_op_id[..., None]
            job_mask = (ops >= job_first) & (ops <= job_last)
            op_job_id = job_mask.argmax(axis=1)
            self.op_flowtime_lb = job_flowtime_lb[
                np.arange(self.number_of_envs)[:, None], op_job_id
            ]

    def set_initial_reward_quality(self):
        if self.objective == ObjectiveFn.MAKESPAN:
            self.init_quality = np.max(self.op_ct_lb, axis=1)
        elif self.objective == ObjectiveFn.AVERAGE_FLOWTIME:
            self.init_quality = np.mean(self.op_ct_lb, axis=1)
        elif self.objective == ObjectiveFn.TOTAL_TARDINESS:
            tardiness_estimates = np.maximum(
                0,
                self.op_ct_lb[
                    np.arange(self.number_of_envs)[:, None], self.job_last_op_id
                ]
                - self.deadlines,
            )
            self.init_quality = np.sum(tardiness_estimates, axis=1)
        elif self.objective == ObjectiveFn.COSTS:
            self.init_quality = np.copy(self.cost_lb)

    def compute_reward(self):
        reward = np.zeros(self.number_of_envs)
        if self.objective == ObjectiveFn.MAKESPAN:
            reward = self.previous_obj_estimate - np.max(self.op_ct_lb, axis=1)
            self.previous_obj_estimate = np.max(self.op_ct_lb, axis=1)
        elif self.objective == ObjectiveFn.AVERAGE_FLOWTIME:
            # The flowtime estimate for each job is equal to the maximum op_ct_lb
            # of the job minus the start time of the first operation if the first operation is scheduled
            # otherwise, the flowtime estimate is the sum of the shortest processing times of the operations
            flowtime_estimates = np.zeros((self.number_of_envs, self.number_of_jobs))
            for env in range(self.number_of_envs):
                for job in range(self.number_of_jobs):
                    if self.job_start_times[env, job] == np.inf:
                        flowtime_estimates[env, job] = np.sum(
                            self.op_min_pt[
                                env,
                                self.job_first_op_id[env, job] : self.job_last_op_id[
                                    env, job
                                ]
                                + 1,
                            ]
                        )
                    else:
                        flowtime_estimates[env, job] = (
                            np.max(
                                self.op_ct_lb[
                                    env,
                                    self.job_first_op_id[
                                        env, job
                                    ] : self.job_last_op_id[env, job] + 1,
                                ]
                            )
                            - self.job_start_times[env, job]
                        )
            reward = self.previous_obj_estimate - np.mean(flowtime_estimates, axis=1)
            self.previous_obj_estimate = np.mean(flowtime_estimates, axis=1)
            # reward = self.previous_obj_estimate - np.sum(flowtime_estimates, axis=1)
        elif self.objective == ObjectiveFn.TOTAL_TARDINESS:
            # The tardiness estimate for each job is equal to the maximum of 0 and
            # the completion the op_ct_lb of the last operation in the job minus the deadline of the job
            tardiness_estimates = np.maximum(
                0,
                self.op_ct_lb[
                    np.arange(self.number_of_envs)[:, None], self.job_last_op_id
                ]
                - self.deadlines,
            )
            reward = self.previous_obj_estimate - np.sum(tardiness_estimates, axis=1)
            self.previous_obj_estimate = np.sum(tardiness_estimates, axis=1)
        elif self.objective == ObjectiveFn.COSTS:
            reward = self.previous_obj_estimate - self.cost_lb
            self.previous_obj_estimate = np.copy(self.cost_lb)
        return reward
