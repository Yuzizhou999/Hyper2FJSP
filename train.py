import os
import random
import sys
import time
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from common_utils import greedy_select_action, sample_action, setup_seed, strToSuffix
from data_utils import (
    CaseGenerator,
    SD2_instance_generator,
    generate_ffsp_instance,
    generate_taillard_jssp_instance,
    load_data_from_files,
)
from enums import ModelArchitecture, ObjectiveFn
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from model.PPO import Memory, PPO_initialize
from params import configs

str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

device = torch.device(configs.device)
configurations = configs


class Trainer:
    def __init__(self, config, multi_objective=False):

        self.n_j = config.n_j
        self.n_m = config.n_m
        self.low = config.low
        self.high = config.high
        self.op_per_job_min = int(0.8 * self.n_m)
        self.op_per_job_max = int(1.2 * self.n_m)
        self.data_source = config.data_source
        self.config = config
        self.max_updates = config.max_updates
        self.reset_env_timestep = config.reset_env_timestep
        self.validate_timestep = config.validate_timestep
        self.num_envs = config.num_envs
        self.deadline_alpha = config.deadline_alpha

        if not os.path.exists(f"./trained_network/{self.data_source}"):
            os.makedirs(f"./trained_network/{self.data_source}")
        if not os.path.exists(f"./train_log/{self.data_source}"):
            os.makedirs(f"./train_log/{self.data_source}")

        if device.type == "cuda":
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        if self.data_source == "SD1":
            self.data_name = f"{self.n_j}x{self.n_m}"
        elif self.data_source == "SD2":
            self.data_name = f"{self.n_j}x{self.n_m}{strToSuffix(config.data_suffix)}"
        elif self.data_source == "JSSP":
            self.data_name = f"{self.n_j}x{self.n_m}_jssp"
        elif self.data_source == "FFSP":
            # For FFSP, create appropriate data name with stages info
            n_stages = getattr(config, "n_stages", min(self.n_m, 4))
            if hasattr(config, "stages_machines"):
                stages_machines = config.stages_machines
            else:
                # Default: distribute machines across stages
                base_machines = max(1, self.n_m // n_stages)
                stages_machines = [base_machines] * n_stages
                # Distribute remaining machines
                remaining = self.n_m - sum(stages_machines)
                for i in range(remaining):
                    stages_machines[i % n_stages] += 1

            stages_str = "_".join(map(str, stages_machines))
            self.data_name = f"{self.n_j}x{n_stages}stages_{stages_str}m_ffsp"

        self.vali_data_path = (
            f"./data/data_train_vali/{self.data_source}/{self.data_name}"
        )
        self.test_data_path = f"./data/{self.data_source}/{self.data_name}"
        self._set_objective_fn(config.objective_fn)
        self._set_model_name(config)

        # seed
        self.seed_train = config.seed_train
        self.seed_test = config.seed_test
        setup_seed(self.seed_train)
        # Make sure model_architecture is compatible with multi-objective setting
        if multi_objective and config.model_architecture == ModelArchitecture.DANIEL:
            print(
                "Warning: Using single-objective model architecture with multi-objective training. Switching to MO_DANIEL_ENC_WEIGHT_INPUT."
            )
            config.model_architecture = ModelArchitecture.MO_DANIEL_ENC_WEIGHT_INPUT
        elif not multi_objective and config.model_architecture.value.startswith("mo_"):
            print(
                "Warning: Using multi-objective model architecture with single-objective training. Switching to DANIEL."
            )
            config.model_architecture = ModelArchitecture.DANIEL

        if multi_objective:
            self.num_preferences_validation = config.num_preferences_validation
        self._set_environments()

        self.ppo = PPO_initialize(multi_objective=multi_objective)
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)
        self.single_value_critic = config.single_value_critic

    def _set_model_name(self, config):
        self.model_name = f"{self.data_name}{strToSuffix(config.model_suffix)}_{config.model_architecture.value}_{config.objective_fn}{'_' + str(config.deadline_alpha) if config.objective_fn == 'total_tardiness' else ''}"

    def _set_objective_fn(self, objective_fn: str):
        self.objective_fn = ObjectiveFn(str(objective_fn).lower())

    def _set_environments(self):
        self.env = FJSPEnvForSameOpNums(
            self.n_j, self.n_m, objective_fn=self.objective_fn
        )
        self.test_data = load_data_from_files(self.test_data_path)
        # validation data set
        vali_data = load_data_from_files(self.vali_data_path)

        if self.data_source == "SD1":
            self.vali_env = FJSPEnvForVariousOpNums(
                self.n_j, self.n_m, objective_fn=self.objective_fn
            )
        elif self.data_source == "SD2":
            self.vali_env = FJSPEnvForSameOpNums(
                self.n_j, self.n_m, objective_fn=self.objective_fn
            )
        elif self.data_source == "JSSP":
            self.vali_env = FJSPEnvForSameOpNums(
                self.n_j, self.n_m, objective_fn=self.objective_fn
            )
        elif self.data_source == "FFSP":
            self.vali_env = FJSPEnvForSameOpNums(
                self.n_j, self.n_m, objective_fn=self.objective_fn
            )

        self.vali_env.set_initial_data(
            vali_data[0], vali_data[1], deadline_alpha=self.deadline_alpha
        )

    def train(self):
        """
        train the model following the config
        """
        self._setup_training()

        for i_update in tqdm(
            range(self.max_updates), file=sys.stdout, desc="progress", colour="blue"
        ):
            state, ep_st = self._initialize_episode(i_update)

            # Use subclass method if available, otherwise use default initialization
            if hasattr(self, "_set_initial_rewards"):
                ep_rewards = self._set_initial_rewards()
            else:
                ep_rewards = np.zeros_like(self.env.init_quality)
                if self.single_value_critic:
                    ep_rewards = self._process_reward(
                        ep_rewards
                    )  # Process reward if using single value critic

            while True:
                # state store
                self.memory.push(state)
                with torch.no_grad():
                    pi_envs, vals_envs = self._forward_pass(state)

                # sample the action
                action_envs, action_logprob_envs = sample_action(pi_envs)

                # state transition
                state, reward, done = self.env.step(actions=action_envs.cpu().numpy())
                if self.single_value_critic:
                    reward = self._process_reward(
                        reward
                    )  # Process reward if using single value critic
                ep_rewards += reward
                reward = torch.from_numpy(reward).to(device)

                # collect the transition
                self.memory.done_seq.append(torch.from_numpy(done).to(device))
                self.memory.reward_seq.append(reward)
                self.memory.action_seq.append(action_envs)
                self.memory.log_probs.append(action_logprob_envs)
                self.memory.val_seq.append(vals_envs.squeeze(1))

                if done.all():
                    break

            loss, v_loss = self._call_update_ppo()
            self.memory.clear_memory()

            mean_rewards_all_env = np.mean(ep_rewards, axis=0)
            mean_makespan_all_env = np.mean(self.env.current_makespan)
            mean_avg_flowtime_all_env = np.mean(self.env.compute_avg_flowtime())
            mean_total_tardiness_all_env = np.mean(self.env.compute_total_tardiness())
            mean_total_earliness_all_env = np.mean(self.env.compute_total_earliness())
            mean_costs_all_env = np.mean(self.env.compute_costs())

            # save the mean rewards of all instances in current training data
            self.log.append([i_update, mean_rewards_all_env])

            # validate the trained model
            if (i_update + 1) % self.validate_timestep == 0:
                if self.data_source == "SD1":
                    vali_result = self.validate_envs_with_various_op_nums(
                        objective_fn=self.objective_fn
                    ).mean()
                else:  # SD2 and JSSP both use same op nums validation
                    vali_result = self.validate_envs_with_same_op_nums(
                        objective_fn=self.objective_fn
                    ).mean()

                if vali_result < self.record:
                    self.save_model()
                    self.record = vali_result

                self.validation_log.append(vali_result)
                self.save_validation_log()
                tqdm.write(
                    f"The validation quality is: {vali_result} (best : {self.record})"
                )

            ep_et = time.time()

            # print the reward, makespan, loss and training time of the current episode
            extra_train_metrics = ""
            if hasattr(self, "ppo") and getattr(
                self.ppo, "last_corner_distill_loss", 0.0
            ):
                extra_train_metrics = (
                    "\t corner distill: {:.8f}\t gate mass: {:.2f}".format(
                        self.ppo.last_corner_distill_loss,
                        getattr(self.ppo, "last_corner_distill_gate_mass", 0.0),
                    )
                )
            tqdm.write(
                "Episode {}\t reward: {}\t makespan: {:.2f}\t avg flowtime: {:.2f}\t total tardiness: {:.2f}\t total earliness: {:.2f}\t total costs: {:.2f}\t Mean_loss: {:.8f}{}\t  training time: {:.2f}".format(
                    i_update + 1,
                    mean_rewards_all_env,
                    mean_makespan_all_env,
                    mean_avg_flowtime_all_env,
                    mean_total_tardiness_all_env,
                    mean_total_earliness_all_env,
                    mean_costs_all_env,
                    loss,
                    extra_train_metrics,
                    ep_et - ep_st,
                )
            )

        self.train_et = time.time()

        # log results
        self.save_training_log()

    def _setup_training(self):
        setup_seed(self.seed_train)
        self.log = []
        self.validation_log = []
        self.record = float("inf")

        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
        print(f"model name :{self.model_name}")
        print(f"vali data :{self.vali_data_path}")
        print("\n")

        self.train_st = time.time()

    def _set_initial_rewards(self):
        ep_rewards = -deepcopy(self.env.init_quality)
        return ep_rewards

    def _initialize_episode(self, i_update):
        ep_st = time.time()
        # resampling the training data
        if i_update % self.reset_env_timestep == 0:
            dataset_job_length, dataset_op_pt = self.sample_training_instances()
            state = self.env.set_initial_data(
                dataset_job_length, dataset_op_pt, deadline_alpha=self.deadline_alpha
            )
        else:
            state = self.env.reset()
        # return dataset_job_length, dataset_op_pt, state, ep_st
        return state, ep_st

    def _forward_pass(self, state):
        pi_envs, vals_envs = self.ppo.policy_old(
            fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
            op_mask=state.op_mask_tensor,  # [sz_b, N, N]
            candidate=state.candidate_tensor,  # [sz_b, J]
            fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
            mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
            comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
            dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
            fea_pairs=state.fea_pairs_tensor,
        )  # [sz_b, J, M]
        return pi_envs, vals_envs

    def _process_reward(self, reward):
        return reward

    def _call_update_ppo(self):
        loss, v_loss = self.ppo.update(self.memory)
        return loss, v_loss

    def save_training_log(self):
        """
        save reward data & validation makespan data (during training) and the entire training time
        """
        file_writing_obj = open(
            f"./train_log/{self.data_source}/" + "reward_" + self.model_name + ".txt",
            "w",
        )
        file_writing_obj.write(str(self.log))

        file_writing_obj1 = open(
            f"./train_log/{self.data_source}/"
            + "valiquality_"
            + self.model_name
            + ".txt",
            "w",
        )
        file_writing_obj1.write(str(self.validation_log))

        file_writing_obj3 = open(f"./train_time.txt", "a")
        file_writing_obj3.write(
            f"model path: ./DANIEL_FJSP/trained_network/{self.data_source}/{self.model_name}\t\ttraining time: "
            f"{round((self.train_et - self.train_st), 2)}\t\t local time: {str_time}\n"
        )

    def save_validation_log(self):
        """
        save the results of validation
        """
        file_writing_obj1 = open(
            f"./train_log/{self.data_source}/"
            + "valiquality_"
            + self.model_name
            + ".txt",
            "w",
        )
        file_writing_obj1.write(str(self.validation_log))

    def sample_training_instances(self):
        """
            sample training instances following the config,
            the sampling process of SD1 data is imported from "songwenas12/fjsp-drl"
        :return: new training instances
        """
        prepare_JobLength = [
            random.randint(self.op_per_job_min, self.op_per_job_max)
            for _ in range(self.n_j)
        ]
        dataset_JobLength = []
        dataset_OpPT = []
        for i in range(self.num_envs):
            if self.data_source == "SD1":
                case = CaseGenerator(
                    self.n_j,
                    self.n_m,
                    self.op_per_job_min,
                    self.op_per_job_max,
                    nums_ope=prepare_JobLength,
                    path="./test",
                    flag_doc=False,
                )
                JobLength, OpPT, _ = case.get_case(i)
            elif self.data_source == "JSSP":
                JobLength, OpPT = generate_taillard_jssp_instance(
                    self.n_j, self.n_m, self.low, self.high
                )
            elif self.data_source == "FFSP":
                # For FFSP, get stages configuration
                n_stages = getattr(self.config, "n_stages", min(self.n_m, 4))
                if hasattr(self.config, "stages_machines"):
                    stages_machines = self.config.stages_machines
                else:
                    # Default: distribute machines across stages
                    base_machines = max(1, self.n_m // n_stages)
                    stages_machines = [base_machines] * n_stages
                    # Distribute remaining machines
                    remaining = self.n_m - sum(stages_machines)
                    for i in range(remaining):
                        stages_machines[i % n_stages] += 1

                JobLength, OpPT = generate_ffsp_instance(
                    self.n_j, stages_machines, self.low, self.high
                )
            else:
                JobLength, OpPT, _ = SD2_instance_generator(config=self.config)
            dataset_JobLength.append(JobLength)
            dataset_OpPT.append(OpPT)

        return dataset_JobLength, dataset_OpPT

    def generate_jssp_validation_data(self, num_validation_instances=50):
        """
        Generate validation data for JSSP instances.

        Args:
            num_validation_instances: Number of validation instances to generate

        Returns:
            Tuple of (JobLength_list, OpPT_list) for validation
        """
        JobLength_list = []
        OpPT_list = []

        # Set a fixed seed for validation data consistency
        current_state = np.random.get_state()
        np.random.seed(self.seed_test)

        for _ in range(num_validation_instances):
            JobLength, OpPT = generate_taillard_jssp_instance(
                self.n_j, self.n_m, self.low, self.high
            )
            JobLength_list.append(JobLength)
            OpPT_list.append(OpPT)

        # Restore the random state
        np.random.set_state(current_state)

        return JobLength_list, OpPT_list

    def validate_envs_with_same_op_nums(self, objective_fn: ObjectiveFn):
        """
            validate the policy using the greedy strategy
            where the validation instances have the same number of operations
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        state = self.vali_env.reset()

        while True:
            with torch.no_grad():
                pi, _ = self.ppo.policy(
                    fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
                    op_mask=state.op_mask_tensor,
                    candidate=state.candidate_tensor,  # [sz_b, J]
                    fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
                    mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
                    comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
                    fea_pairs=state.fea_pairs_tensor,
                )  # [sz_b, J, M]

            action = greedy_select_action(pi)
            state, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        if objective_fn == ObjectiveFn.MAKESPAN:
            return self.vali_env.current_makespan
        elif objective_fn == ObjectiveFn.AVERAGE_FLOWTIME:
            return self.vali_env.compute_avg_flowtime()
        elif objective_fn == ObjectiveFn.TOTAL_TARDINESS:
            return self.vali_env.compute_total_tardiness()
        elif objective_fn == ObjectiveFn.NEGATIVE_MAKESPAN:
            return -self.vali_env.current_makespan
        elif objective_fn == ObjectiveFn.COSTS:
            return self.vali_env.compute_costs()

    def validate_envs_with_various_op_nums(self, objective_fn: ObjectiveFn):
        """
            validate the policy using the greedy strategy
            where the validation instances have various number of operations
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        state = self.vali_env.reset()

        while True:
            with torch.no_grad():
                batch_idx = ~torch.from_numpy(self.vali_env.done_flag)
                pi, _ = self.ppo.policy(
                    fea_j=state.fea_j_tensor[batch_idx],  # [sz_b, N, 8]
                    op_mask=state.op_mask_tensor[batch_idx],
                    candidate=state.candidate_tensor[batch_idx],  # [sz_b, J]
                    fea_m=state.fea_m_tensor[batch_idx],  # [sz_b, M, 6]
                    mch_mask=state.mch_mask_tensor[batch_idx],  # [sz_b, M, M]
                    comp_idx=state.comp_idx_tensor[batch_idx],  # [sz_b, M, M, J]
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor[
                        batch_idx
                    ],  # [sz_b, J, M]
                    fea_pairs=state.fea_pairs_tensor[batch_idx],
                )  # [sz_b, J, M]

            action = greedy_select_action(pi)
            state, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        if objective_fn == ObjectiveFn.MAKESPAN:
            return self.vali_env.current_makespan
        elif objective_fn == ObjectiveFn.AVERAGE_FLOWTIME:
            return self.vali_env.compute_avg_flowtime()
        elif objective_fn == ObjectiveFn.TOTAL_TARDINESS:
            return self.vali_env.compute_total_tardiness()
        elif objective_fn == ObjectiveFn.NEGATIVE_MAKESPAN:
            return -self.vali_env.current_makespan
        elif objective_fn == ObjectiveFn.COSTS:
            return self.vali_env.compute_costs()

    def save_model(self):
        """
        save the model
        """
        torch.save(
            self.ppo.policy.state_dict(),
            f"./trained_network/{self.data_source}/{self.model_name}.pth",
        )

    def load_model(self):
        """
        load the trained model
        """
        model_path = f"./trained_network/{self.data_source}/{self.model_name}.pth"
        self.ppo.policy.load_state_dict(torch.load(model_path, map_location="cuda"))


def main():
    trainer = Trainer(configurations)
    trainer.train()


if __name__ == "__main__":
    main()
