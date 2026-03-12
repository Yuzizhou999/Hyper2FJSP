import argparse
from enums import ModelArchitecture, ObjectiveFn


def str2bool(v):
    """
        transform string value to bool value
    :param v: a string input
    :return: the bool value
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')



parser = argparse.ArgumentParser(description='Arguments for DANIEL_FJSP')
# args for device
parser.add_argument('--device', type=str, default='cuda', help='Device name')
parser.add_argument('--device_id', type=str, default='0', help='Device id')

# args for file_name

parser.add_argument('--model_suffix', type=str, default='', help='Suffix of the model')
parser.add_argument('--data_suffix', type=str, default='mix', help='Suffix of the data')

# args for AutoExperiment
parser.add_argument('--cover_flag', type=str2bool, default=True, help='Whether covering test results of the model')
parser.add_argument('--cover_data_flag', type=str2bool, default=False, help='Whether covering the generated data')
parser.add_argument('--cover_heu_flag', type=str2bool, default=True,
                    help='Whether covering test results of heuristics')
parser.add_argument('--cover_train_flag', type=str2bool, default=True, help='Whether covering the trained model')

# args for data load
parser.add_argument('--model_source', type=str, default='SD2', help='Suffix of the data that model trained on')
parser.add_argument('--data_source', type=str, default='SD2', help='Suffix of test data')

# args for SD2 data generation
parser.add_argument('--op_per_job', type=float, default=0,
                    help='Number of operations per job, default 0, means the number equals m')
parser.add_argument('--op_per_mch_min', type=int, default=1,
                    help='Minimum number of compatible machines for each operation')
parser.add_argument('--op_per_mch_max', type=int, default=5,
                    help='Maximum number of compatible machines for each operation')
parser.add_argument('--data_size', type=int, default=100, help='The number of instances for data generation')
parser.add_argument('--data_type', type=str, default="test", help='Generated data type (test/vali)')

# args for FFSP data generation
parser.add_argument('--n_stages', type=int, default=4, help='Number of stages for FFSP instances')
parser.add_argument('--stages_machines', nargs='+', type=int, default=[3, 3, 3, 3], 
                    help='Number of machines per stage for FFSP instances (e.g., --stages_machines 3 3 3 3)')


# args for testData to excel
parser.add_argument('--sort_flag', type=str2bool, default=True,
                    help='Whether sorting the printed results by the makespan')

# args for or-tools
parser.add_argument('--max_solve_time', type=int, default=300, help='The maximum solving time of OR-Tools')

# args for seed
parser.add_argument('--seed_datagen', type=int, default=200, help='Seed for data generation')
parser.add_argument('--seed_train_vali_datagen', type=int, default=100, help='Seed for generate validation data')
parser.add_argument('--seed_train', type=int, default=300, help='Seed for training')
parser.add_argument('--seed_test', type=int, default=50, help='Seed for testing heuristics')
# args for tricks

# args for env
parser.add_argument('--n_j', type=int, default=10, help='Number of jobs of the instance')
parser.add_argument('--n_m', type=int, default=5, help='Number of machines of the instance')
parser.add_argument('--n_op', type=int, default=50, help='Number of operations of the instance')
parser.add_argument('--low', type=int, default=1, help='Lower Bound of processing time(PT)')
parser.add_argument('--high', type=int, default=99, help='Upper Bound of processing time')

# args for network
parser.add_argument('--fea_j_input_dim', type=int, default=11, help='Dimension of operation raw feature vectors')
parser.add_argument('--fea_m_input_dim', type=int, default=8, help='Dimension of machine raw feature vectors')

parser.add_argument('--dropout_prob', type=float, default=0.0, help='Dropout rate (1 - keep probability).')

parser.add_argument('--num_heads_OAB', nargs='+', type=int, default=[4, 4],
                    help='Number of attention head of operation message attention block')
parser.add_argument('--num_heads_MAB', nargs='+', type=int, default=[4, 4],
                    help='Number of attention head of machine message attention block')
parser.add_argument('--layer_fea_output_dim', nargs='+', type=int, default=[32, 8],
                    help='Output dimension of the DAN layers')

parser.add_argument('--num_mlp_layers_actor', type=int, default=3, help='Number of layers in Actor network')
parser.add_argument('--hidden_dim_actor', type=int, default=64, help='Hidden dimension of Actor network')
parser.add_argument('--num_mlp_layers_critic', type=int, default=3, help='Number of layers in Critic network')
parser.add_argument('--hidden_dim_critic', type=int, default=64, help='Hidden dimension of Critic network')

# Add parameter for gamma/beta transformation
parser.add_argument('--use_gamma_beta', type=str2bool, default=True, 
                   help='Whether to use gamma/beta transformation in MO attention blocks')


# args for PPO Algorithm
parser.add_argument('--num_envs', type=int, default=20, help='Batch size for training environments')
parser.add_argument('--max_updates', type=int, default=1000, help='No. of episodes of each env for training')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')

parser.add_argument('--gamma', type=float, default=1, help='Discount factor used in training')
parser.add_argument('--k_epochs', type=int, default=4, help='Update frequency of each episode')
parser.add_argument('--eps_clip', type=float, default=0.2, help='Clip parameter')
parser.add_argument('--vloss_coef', type=float, default=0.5, help='Critic loss coefficient')
parser.add_argument('--ploss_coef', type=float, default=1, help='Policy loss coefficient')
parser.add_argument('--entloss_coef', type=float, default=0.01, help='Entropy loss coefficient')
parser.add_argument('--tau', type=float, default=0, help='Policy soft update coefficient')
parser.add_argument('--gae_lambda', type=float, default=0.98, help='GAE parameter')

# args for training
parser.add_argument('--train_size', type=str, default="10x5", help='Size of training instances')
parser.add_argument('--validate_timestep', type=int, default=40, help='Interval for validation and data log')
parser.add_argument('--reset_env_timestep', type=int, default=20, help='Interval for reseting the environment')
parser.add_argument('--minibatch_size', type=int, default=1024, help='Batch size for computing the gradient')
parser.add_argument('--max_grad_norm', type=float, default=0.5,
                    help='Maximum gradient norm for PPO updates; set <= 0 to disable clipping')

# args for test
parser.add_argument('--test_data', nargs='+', default=['20x10+mix'], help='List of data for testing')
parser.add_argument('--test_mode', type=str2bool, default=False, help='Whether using the sampling strategy in testing')
parser.add_argument('--sample_times', type=int, default=10, help='Sampling times for the sampling strategy')
parser.add_argument('--num_sampling_cycles', type=int, default=1, help='Number of sampling cycles to run (results stored per cycle)')
parser.add_argument('--test_model', nargs='+', default=['20x10+mix_MO'], help='List of model for testing')
parser.add_argument('--test_method', nargs='+', default=[], help='List of heuristic methods for testing')
# new args
parser.add_argument('--objective_fn', nargs='+', default=['total_tardiness', 'makespan'], help='Objective function for the environment')
parser.add_argument('--deadline_alpha', type=float, default=0.5, help='Deadline alpha for the environment')

parser.add_argument('--num_preferences_validation', type=int, default=51, help='Number of preferences for validation')
parser.add_argument('--num_preferences_test', type=int, default=101, help='Number of preferences for testing')

# Add model architecture parameter with hardcoded possible values in the help description
parser.add_argument(
    '--model_architecture',
    type=str,
    default='mo_daniel_conditional',
    help=(
        "Model architecture to use. Allowed values match ModelArchitecture enum: "
        "daniel, mo_daniel_enc_weight_input, mo_daniel_conditional_op_and_mch, "
        "mo_daniel_conditional_op_and_mch_fea_input, hyper_daniel"
    )
)

# Hypernetwork parameters
parser.add_argument('--hyper_hidden_dim', type=int, default=256, 
                    help='Hidden dimension for hypernetwork (used with hyper_daniel architecture)')
parser.add_argument('--hyper_embd_dim', type=int, default=2,
                    help='Embedding dimension for hypernetwork parameter generation (used with hyper_daniel architecture)')
parser.add_argument('--hyper_use_instance_features', type=str2bool, default=False,
                    help='Whether HYPER conditions the hypernetwork on pooled instance features in addition to preferences')


parser.add_argument('--single_value_critic', type=str2bool, default=False, help='Whether to use a single value critic for PPO')
parser.add_argument('--use_simple_reward', type=str2bool, default=False,
                    help='Use simple actual-value rewards during training (based on currently scheduled operations only). Default: False (uses lower bound estimates)')
parser.add_argument('--use_lb_features', type=str2bool, default=True,
                    help='Use lower bound features (op_ct_lb, op_lateness_lb, op_flowtime_lb) in operation representations. Default: True')
parser.add_argument('--task_id', type=int, default=0, help='Task ID for distributed training')

configs = parser.parse_args()

# Convert string model architecture to enum
try:
    configs.model_architecture = ModelArchitecture.from_string(configs.model_architecture)
except ValueError as e:
    print(f"Warning: {e}. Using default model architecture (DANIEL).")
    configs.model_architecture = ModelArchitecture.DANIEL
    
# Set default feature dimensions
base_fea_j_input_dim = 9  # Base features (always included):
                           # scheduled_flag, op_min_pt, pt_span, op_mean_pt, waiting_time,
                           # remain_work, left_op_nums, job_remain_work, available_mchs

# Add op_ct_lb if using lower bound features
if configs.use_lb_features:
    base_fea_j_input_dim += 1  # op_ct_lb

# Parse objective functions and adjust dimensions based on them
objective_fn_list = []
if isinstance(configs.objective_fn, str):
    objective_fn_list = [ObjectiveFn(fn.strip().lower()) for fn in configs.objective_fn.split(',')]
elif isinstance(configs.objective_fn, list):
    objective_fn_list = [ObjectiveFn(fn.strip().lower()) for fn in configs.objective_fn]

# Dynamically adjust fea_j_input_dim based on objectives
fea_j_input_dim = base_fea_j_input_dim

# Add objective-specific LB features only if use_lb_features is True
# op_lateness_lb is used for both tardiness and earliness objectives
if (ObjectiveFn.TOTAL_TARDINESS in objective_fn_list or ObjectiveFn.TOTAL_EARLINESS in objective_fn_list):
    if configs.use_lb_features:
        fea_j_input_dim += 1  # op_lateness_lb

if ObjectiveFn.AVERAGE_FLOWTIME in objective_fn_list:
    if configs.use_lb_features:
        fea_j_input_dim += 1  # op_flowtime_lb

# Set the computed dimensions in the configs
configs.fea_j_input_dim = fea_j_input_dim
