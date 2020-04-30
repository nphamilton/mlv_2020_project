%% Compute the reach set of the unsafe controller
load ddpg_rs_8.mat
[P1, reachTime1, sim_traces1] = verify_pendulum(W, b);

%% Plot the reach set of the unsafe controller
% plot output (cos(theta) x[1] and sin(theta) x[2])
maps = [1 0 0; 0 1 0];

Pos_Vel_ReachSet1 = [];
for i=length(P1):-1:2
    Pos_Vel_ReachSet1 = [Pos_Vel_ReachSet1 P1(i).affineMap(maps, [])];
end

% full plot
figure;
Star.plotBoxes_2D(Pos_Vel_ReachSet1, 1, 2, 'blue');
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
Star.plotBoxes_2D(Pos_Vel_ReachSet1, 1, 2, 'blue');
Star.plotBoxes_2D(P1(1).affineMap(maps, []), 1, 2, 'black');
xlim([cosd(16), 1])
ylim([sind(-16), sind(16)])
xline(cosd(15), 'r', 'LineWidth', 2.0);
yline(sind(-15), 'r', 'LineWidth', 2.0);
yline(sind(15), 'r', 'LineWidth', 2.0);
title('DDPG Reachability')
for i=1:length(sim_traces1)
    plot(sim_traces1{1,i}(1, :),sim_traces1{1,i}(2, :))
end

%% Compute the reach set of the safer controller
load ddpg_c_rs_1964.mat
[P2, reachTime2, sim_traces2] = verify_pendulum(W, b);

%% Plot the reach set of the safe controller
% plot output (cos(theta) x[1] and sin(theta) x[2])
maps = [1 0 0; 0 1 0];

Pos_Vel_ReachSet2 = [];
for i=length(P2):-1:2
    Pos_Vel_ReachSet2 = [Pos_Vel_ReachSet2 P2(i).affineMap(maps, [])];
end

% full plot
figure;
Star.plotBoxes_2D(Pos_Vel_ReachSet2, 1, 2, 'blue');
Star.plotBoxes_2D(P2(1).affineMap(maps, []), 1, 2, 'black');
xline(cosd(15), 'r');
yline(sind(-15), 'r');
yline(sind(15), 'r');
title('DDPG-C Reachability')
for i=1:length(sim_traces2)
    plot(sim_traces2{1,i}(1, :),sim_traces2{1,i}(2, :), '-')
end

% Zoomed in on the safe region
figure;
Star.plotBoxes_2D(Pos_Vel_ReachSet2, 1, 2, 'blue');
Star.plotBoxes_2D(P2(1).affineMap(maps, []), 1, 2, 'black');
xlim([cosd(16), 1])
ylim([sind(-16), sind(16)])
xline(cosd(15), 'r', 'LineWidth', 2.0);
yline(sind(-15), 'r', 'LineWidth', 2.0);
yline(sind(15), 'r', 'LineWidth', 2.0);
title('DDPG-C Reachability')
for i=1:length(sim_traces2)
    plot(sim_traces2{1,i}(1, :),sim_traces2{1,i}(2, :), '-')
end

%% Compute the reach set of the safe controller without the barriers
load cbf_n_rs_1754.mat
[P3, reachTime3, sim_traces3] = verify_pendulum(W, b);

%% Plot the reach set of the safe controller
% plot output (cos(theta) x[1] and sin(theta) x[2])
maps = [1 0 0; 0 1 0];

Pos_Vel_ReachSet3 = [];
for i=length(P3):-1:2
    Pos_Vel_ReachSet3 = [Pos_Vel_ReachSet3 P3(i).affineMap(maps, [])];
end

% full plot
figure;
Star.plotBoxes_2D(Pos_Vel_ReachSet3, 1, 2, 'blue');
Star.plotBoxes_2D(P3(1).affineMap(maps, []), 1, 2, 'black');
xline(cosd(15), 'r');
yline(sind(-15), 'r');
yline(sind(15), 'r');
title('CBF-N Reachability')
for i=1:length(sim_traces3)
    plot(sim_traces3{1,i}(1, :),sim_traces3{1,i}(2, :), '-')
end

% Zoomed in on the safe region
figure;
Star.plotBoxes_2D(Pos_Vel_ReachSet3, 1, 2, 'blue');
Star.plotBoxes_2D(P3(1).affineMap(maps, []), 1, 2, 'black');
xlim([cosd(16), 1])
ylim([sind(-16), sind(16)])
xline(cosd(15), 'r', 'LineWidth', 2.0);
yline(sind(-15), 'r', 'LineWidth', 2.0);
yline(sind(15), 'r', 'LineWidth', 2.0);
title('CBF-N Reachability')
for i=1:length(sim_traces3)
    plot(sim_traces3{1,i}(1, :),sim_traces3{1,i}(2, :), '-')
end

%% Display the computation times
disp('DDPG took this many seconds to verify: ')
disp(reachTime1)
disp('DDPG-C took this many seconds to verify: ')
disp(reachTime2)
disp('CBF-N took this many seconds to verify: ')
disp(reachTime3)