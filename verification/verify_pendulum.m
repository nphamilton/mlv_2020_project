function [P, reachTime, sim_traces] = verify_pendulum(W, b)


%% Create plant model
num_state_vars = 3;
num_inputs = 1;
Ts = 0.05;
outputMat = [1, 0, 0;
             0, 1, 0;
             0, 0, 1];

plant = DNonLinearODE(num_state_vars, num_inputs, @pendulum_dynamics, Ts, outputMat);

%% Create FFNN model
% load(nn_model_path);

L1 = LayerS(double(W{1,1}'), double(b{1,1}'), 'poslin');
L2 = LayerS(double(W{1,2}'), double(b{1,2}'), 'poslin');
out_layer = LayerS(double(W{1,3}'), double(b{1,3}'), 'tansig');
out_scale = LayerS(double(2.0), double(0.0), 'purelin');

% feedforward neural network controller
controller = FFNNS([L1 L2 out_layer out_scale]); 
feedbackMap = [0]; % feedback map, y[k]

%% the neural network control system
ncs = DNonlinearNNCS(controller, plant, feedbackMap); 

%% Compute reachability
% initial set of state of the plant x = [-0.1 <= x[1] <= -0.05, 0.85 <= x[2] <= 0.9, x[3] = 0; x[4] = 0]
% lb = [cosd(-15); sind(-15); -1.0];
% % lb = [cosd(0), sind(0), 
% ub = [cosd(0); sind(15); 1.0];
% lb = [0.95; 0.25; 0.98];
% ub = [0.96592582628; 0.2588190451; 1.0];
% lb = [cosd(90); sind(89); 7.9];
% ub = [cosd(89); sind(90); 8.0];
lb = [cosd(5.0); sind(-5.0); -0.01];
ub = [cosd(0.0); sind(5.0); 0.01];
init_set = Star(lb, ub);
n_steps = 10;

reachPRM.init_set = init_set;
reachPRM.ref_input = [];
reachPRM.numCores = 1;
reachPRM.numSteps = n_steps;
reachPRM.reachMethod = 'approx-star';

[P, reachTime] = ncs.reach(reachPRM);

[sim_time, sim_traces, control_traces, sampled_init_states, ...
    sampled_ref_inputs] = ...
    ncs.sample(n_steps, init_set.getBox(), [], 100);

end