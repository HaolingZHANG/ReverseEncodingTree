# neat-python configuration for the LunarLander-v2 environment on OpenAI Gym

[NEAT]
pop_size              = 20
fitness_criterion     = max
fitness_threshold     = -0.6
reset_on_extinction   = 0

[GlobalGenome]
# node activation options
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 1.0

# connection add/remove rates
conn_add_prob           = 0.9
conn_delete_prob        = 0.2

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True

# node add/remove rates
node_add_prob           = 0.9
node_delete_prob        = 0.2

# network parameters
num_hidden              = 0
num_inputs              = 2
num_outputs             = 3
max_node_num            = 6

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30.
weight_min_value        = -30.
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[StrongSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 4

[Reproduction]
init_distance    = 2
min_distance     = 0.1
correlation_rate = 0.5
search_count     = 30