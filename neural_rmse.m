clc; clear; close all;

% material properties
E = 20e9;        % Young's modulus (Pa)
nu = 0.27;        % Poisson's ratio
rho = 751;        % Density (kg/m³)

% Compute Lame parameters
lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
mu = E / (2 * (1 + nu));

% Compute wave speed
c = sqrt((E * (1 - nu)) / ((1 + nu) * (1 - 2 * nu) * rho));

% Generate training data (collocation points)
Nx = 20; Ny = 20; Nt = 20; % Number of points in space (x, y) and time (t)
x = linspace(0, 1, Nx)'; % Spatial points in x
y = linspace(0, 1, Ny)'; % Spatial points in y
t = linspace(0, 1, Nt)'; % Time points
[X, Y, T] = meshgrid(x, y, t);
X_train = [X(:), Y(:), T(:)]; % Convert to Nx*Ny*Nt x 3 format

% Define initial conditions Ux(x,y,0) = sin(pi*x)*sin(pi*y)
U0_x = sin(pi * X(:)) .* sin(pi *c* Y(:));
U0_y = cos(pi * X(:)) .* sin(pi *c* Y(:));
U0 = [U0_x, U0_y];

% Define neural network architecture
layers = [
    featureInputLayer(3)      % Input: (x, y, t)
    fullyConnectedLayer(64)   % Hidden layer 1
    tanhLayer                 % Activation function
    fullyConnectedLayer(64)   % Hidden layer 2
    tanhLayer                 % Activation function
    fullyConnectedLayer(64)   % Hidden layer 3
    tanhLayer                 % Activation function
    fullyConnectedLayer(64)   % Hidden layer 4
    tanhLayer                 % Activation function
    fullyConnectedLayer(2)    % Output: (u_x, u_y)
    regressionLayer
];

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 1000, ...
    'InitialLearnRate', 0.001, ...
    'Plots', 'training-progress');

% Train neural network
net = trainNetwork(X_train, U0, layers, options);

% Predict displacement solution using trained PINN
U_pred = predict(net, X_train); % Shape (Nx*Ny*Nt x 2)
U_pred_x = reshape(U_pred(:,1), Nx, Ny, Nt);
U_pred_y = reshape(U_pred(:,2), Nx, Ny, Nt);

% Calculate RMSE errors
rmse_x = sqrt(mean((U_pred_x(:) - U0_x(:)).^2));
rmse_y = sqrt(mean((U_pred_y(:) - U0_y(:)).^2));

% Calculate RMSE as percentage of max amplitude
max_amp_x = max(abs(U0_x(:)));
max_amp_y = max(abs(U0_y(:)));
rmse_pct_x = (rmse_x / max_amp_x) * 100;
rmse_pct_y = (rmse_y / max_amp_y) * 100;

% Plot results with error annotations
hFig = figure('Position', [100, 100, 1200, 500]);

subplot(1,2,1);
surf(x, y, U_pred_x(:,:,end));
xlabel('X'); ylabel('Y'); zlabel('Displacement U_x');
title(sprintf('PINN Solution for u_x(x,y,t)\nRMSE: %.2e (%.1f%%)', rmse_x, rmse_pct_x));
colormap jet; shading interp;

subplot(1,2,2);
surf(x, y, U_pred_y(:,:,end));
xlabel('X'); ylabel('Y'); zlabel('Displacement U_y');
title(sprintf('PINN Solution for u_y(x,y,t)\nRMSE: %.2e (%.1f%%)', rmse_y, rmse_pct_y));
colormap jet; shading interp;

% Display errors in command window
fprintf('\nError Analysis:\n');
fprintf('U_x RMSE: %.4e (%.2f%% of max amplitude)\n', rmse_x, rmse_pct_x);
fprintf('U_y RMSE: %.4e (%.2f%% of max amplitude)\n', rmse_y, rmse_pct_y);

% FUNCTION DEFINITIONS
function loss = elastodynamicLoss(X_train, net, lambda, mu, rho)
    x = X_train(:,1); y = X_train(:,2); t = X_train(:,3);
    
    % Compute neural network prediction
    U_pred = predict(net, X_train);
    u_x = U_pred(:,1); u_y = U_pred(:,2);
    
    % Compute gradients
    [u_x_t, u_x_x, u_x_y] = dlgradient(sum(u_x), X_train);
    [u_y_t, u_y_x, u_y_y] = dlgradient(sum(u_y), X_train);
    [u_x_tt, ~, ~] = dlgradient(sum(u_x_t), X_train);
    [u_y_tt, ~, ~] = dlgradient(sum(u_y_t), X_train);
    
    % Stress components
    sigma_xx = lambda * (u_x_x + u_y_y) + 2 * mu * u_x_x;
    sigma_yy = lambda * (u_x_x + u_y_y) + 2 * mu * u_y_y;
    sigma_xy = mu * (u_x_y + u_y_x);
    
    % Compute PDE residual loss
    residual_x = (sigma_xx + sigma_xy - rho * u_x_tt);
    residual_y = (sigma_xy + sigma_yy - rho * u_y_tt);
    loss = mse(residual_x, zeros(size(residual_x))) + mse(residual_y, zeros(size(residual_y)));
end