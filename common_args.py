
def add_dataset_args(parser):
    parser.add_argument("--envs", type=int, required=False,
                        default=100000, help="Envs")
    parser.add_argument("--envs_eval", type=int, required=False,
                        default=200, help="Eval Envs")
    parser.add_argument("--hists", type=int, required=False,
                        default=1, help="Histories")
    parser.add_argument("--samples", type=int,
                        required=False, default=1, help="Samples")
    parser.add_argument("--H", type=int, required=False,
                        default=300, help="Context horizon")
    parser.add_argument("--dim", type=int, required=False,
                        default=5, help="Dimension")
    parser.add_argument("--lin_d", type=int, required=False,
                        default=2, help="Linear feature dimension")

    parser.add_argument("--var", type=float, required=False,
                        default=0.1, help="Bandit arm variance")
    parser.add_argument("--cov", type=float, required=False,
                        default=0.0, help="Coverage of optimal arm")
    parser.add_argument("--rdm_fix_ratio", type = list, required = False, default = [1.0, 0.0], help = "Ratio of random-arm and fixed-arm trajectories")

    parser.add_argument("--env", type=str, required=False, default = "bandit", help="Environment")
    parser.add_argument("--env_id_start", type=int, required=False,
                        default=-1, help="Start index of envs to sample")
    parser.add_argument("--env_id_end", type=int, required=False,
                        default=-1, help="End index of envs to sample")
    
    parser.add_argument("--merged_data?", type = bool, required = False, default = False, help = "Whether to use data from different envs")
    parser.add_argument("--env_names", type=list, required=False, default=["cgbandit", "bandit"], help="environment for merged dataset")
    parser.add_argument("--ratio", type=list, required=False, default=[1.0, 0.0], help="ratio of environments for merged dataset")


def add_model_args(parser):
    parser.add_argument("--embd", type=int, required=False,
                        default=32, help="Embedding size")
    parser.add_argument("--head", type=int, required=False,
                        default=1, help="Number of heads")
    parser.add_argument("--layer", type=int, required=False,
                        default=4, help="Number of layers")
    parser.add_argument("--context_len", type=int, required=False, default = 100, help = "Context length of transformer")
    parser.add_argument("--lr", type=float, required=False,
                        default=1e-3, help="Learning Rate")
    parser.add_argument("--dropout", type=float,
                        required=False, default=0.0, help="Dropout")
    parser.add_argument("--class", type=str, required=False,default='dpt', help="Model_Type")
    parser.add_argument('--shuffle', required = False, default=False, action='store_true')
    parser.add_argument('--context_type', required = False, default = 'ground_truth', help = "Context type")
    parser.add_argument('--intermediate_size', required = False, default=1024, help = "Intermediate size")
    parser.add_argument("--exploration_rate", type=float, required=False,
                        default=0.0, help="Exploration rate")


def add_train_args(parser):
    parser.add_argument("--num_epochs", type=int, required=False,
                        default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, required=False,
                        default=64, help="Batch size")


def add_eval_args(parser):
    parser.add_argument("--epoch", type=int, required=False,
                        default=100, help="Epoch to evaluate")
    parser.add_argument("--test_cov", type=float,
                        required=False, default=-1.0,
                        help="Test coverage (for bandit)")
    parser.add_argument("--hor", type=int, required=False,
                        default=-1, help="Episode horizon (for mdp)")
    parser.add_argument("--n_eval", type=int, required=False,
                        default=200, help="Number of eval trajectories")
    parser.add_argument("--save_video", default=False, action='store_true')

