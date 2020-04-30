function [next_x] = pendulum_dynamics(t, x, u, dt)

%% Constants
max_speed = 8.0;
g = 10.0;
m = 1.0;
l = 1.0;
A = -3*g/(2*l);
B = 3./(m*l^2);

%% Break apart the current state
cos_theta = x(1);
sin_theta = x(2);
theta_dot = x(3);

%% Compute the next state
% theta = atan(sin_theta / cos_theta);
theta = asin(sin_theta);

newthdot = theta_dot + (A * sin(theta + pi) + B*u) * dt;
% newthdot = min(max(newthdot, -max_speed), max_speed);

new_theta = theta + newthdot*dt;

%% Separate the new state into the state variables
next_x = [cos(new_theta); sin(new_theta); newthdot];

end