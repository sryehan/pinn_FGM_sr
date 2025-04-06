clc;
clear;
close all;
% Define parameters
H = 0.1;            % Length of the domain (m)
NT = 1;        % Number of time steps


E = 20e9;        % Young's modulus (Pa)
nu = 0.27;        % Poisson's ratio
rho = 751;        % Density (kg/m³)
c = sqrt((E * (1 - nu)) / ((1 + nu) * (1 - 2 * nu) * rho))



n_terms = 10; % Number of terms in the series

% Discretization
x = linspace(0, H, 100);
t = linspace(0, NT, 100); % Time range
[X, T] = meshgrid(x, t);

% Velocity function
F_x = @(x) 100000* pi * sin(3* pi * x / H);

% Initialize solution matrices
U_fixed_fixed = zeros(size(X));
U_free_free = zeros(size(X));

% Compute K_f coefficients
for f = 1:n_terms
    K_f = (2 / (f * pi * c)) * integral(@(x) F_x(x) .* sin(f * pi * x / H), 0, H);
    U_fixed_fixed = U_fixed_fixed + K_f * cos(f * pi * X / H) .* sin(f * pi * c * T / H);
    U_free_free = U_free_free + K_f * sin(f * pi * X / H) .* sin(f * pi * c * T / H);
end


% Plot Fixed-Fixed boundary condition
subplot(1,2,1);
surf(X, T, U_fixed_fixed);
shading interp;
colormap jet;
xlabel('x'); ylabel('t'); zlabel('U(x,t)');
title('Fixed-Fixed Boundary Condition');
colorbar;
grid off;

% Plot Free-Free boundary condition
subplot(1,2,2);
surf(X, T, U_free_free);
shading interp;
colormap jet;
xlabel('x'); ylabel('t'); zlabel('U(x,t)');
title('Free-Free Boundary Condition');
colorbar;
grid off;


