function [] = verify_pendulum(W, b)


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

L1 = LayerS(W{1,1}', b{1,1}', 'poslin');
L2 = LayerS(W{1,2}', b{1,2}', 'poslin');
out_layer = LayerS(W{1,3}', b{1,3}', 'tansig');
out_scale = LayerS(15, 0, 'purelin');

% feedforward neural network controller
controller = FFNNS([L1 L2 out_layer out_scale]); 
feedbackMap = [0]; % feedback map, y[k]

%% the neural network control system
ncs = DNonlinearNNCS(controller, plant, feedbackMap); 

%% Compute reachability
% initial set of state of the plant x = [-0.1 <= x[1] <= -0.05, 0.85 <= x[2] <= 0.9, x[3] = 0; x[4] = 0]
lb = [0.0; -0.2588190451; -1.0];
ub = [0.96592582628; 0.2588190451; 1.0];
init_set = Star(lb, ub);

reachPRM.init_set = init_set;
reachPRM.ref_input = [];
reachPRM.numCores = 1;
reachPRM.numSteps = 10;
reachPRM.reachMethod = 'approx-star';

[P, reachTime] = ncs.reach(reachPRM);

% plot output (position x[1] and velocity x[2])
maps = [1 0 0; 0 1 0];

Pos_Vel_ReachSet = [];
for i=1:length(P)
    Pos_Vel_ReachSet = [Pos_Vel_ReachSet P.affineMap(maps)];
end

figure;
Pos_Vel_ReachSet.plot;

end