clear all;
close all;

%% Compute the reach set of the unsafe controller
load ddpg_rs_8.mat
[P1, reachTime1, sim_traces1] = verify_pendulum(W, b);

%% Plot the reach set of the unsafe controller
% plot output (cos(theta) x[1] and sin(theta) x[2])
maps = [1 0 0; 0 1 0];

Pos_Vel_ReachSet = [];
for i=length(P1):-1:2
    Pos_Vel_ReachSet = [Pos_Vel_ReachSet P1(i).affineMap(maps, [])];
end

% full plot
figure;
Star.plotBoxes_2D(Pos_Vel_ReachSet, 1, 2, 'blue');
Star.plotBoxes_2D(P1(1).affineMap(maps, []), 1, 2, 'black');
xline(cosd(15), 'r');
yline(sind(-15), 'r');
yline(sind(15), 'r');
title('DDPG Reachability')
for i=1:length(sim_traces1)
    plot(sim_traces1{1,i}(1, :),sim_traces1{1,i}(2, :))
end

% Zoomed in on the safe region
figure;
Star.plotBoxes_2D(Pos_Vel_ReachSet, 1, 2, 'blue');
Star.plotBoxes_2D(P1(1).affineMap(maps, []), 1, 2, 'black');
xlim([cosd(16), 1])
ylim([sind(-16), sind(16)])
xline(cosd(15), 'r');
yline(sind(-15), 'r');
yline(sind(15), 'r');
title('DDPG Reachability')
for i=1:length(sim_traces1)
    plot(sim_traces1{1,i}(1, :),sim_traces1{1,i}(2, :))
end

%% Compute the reach set of the safe controller
load ddpg_rs_8.mat
[P2, reachTime2, sim_traces2] = verify_pendulum(W, b);

%% Plot the reach set of the safe controller
% plot output (cos(theta) x[1] and sin(theta) x[2])
maps = [1 0 0; 0 1 0];

Pos_Vel_ReachSet = [];
for i=length(P2):-1:2
    Pos_Vel_ReachSet = [Pos_Vel_ReachSet P2(i).affineMap(maps, [])];
end

% full plot
figure;
Star.plotBoxes_2D(Pos_Vel_ReachSet, 1, 2, 'blue');
Star.plotBoxes_2D(P2(1).affineMap(maps, []), 1, 2, 'black');
xline(cosd(15), 'r');
yline(sind(-15), 'r');
yline(sind(15), 'r');
title('DDPG Reachability')
for i=1:length(sim_traces2)
    plot(sim_traces2{1,i}(1, :),sim_traces2{1,i}(2, :), '-')
end

% Zoomed in on the safe region
figure;
Star.plotBoxes_2D(Pos_Vel_ReachSet, 1, 2, 'blue');
Star.plotBoxes_2D(P2(1).affineMap(maps, []), 1, 2, 'black');
xlim([cosd(16), 1])
ylim([sind(-16), sind(16)])
xline(cosd(15), 'r');
yline(sind(-15), 'r');
yline(sind(15), 'r');
title('DDPG Reachability')
for i=1:length(sim_traces2)
    plot(sim_traces2{1,i}(1, :),sim_traces2{1,i}(2, :), '-')
end