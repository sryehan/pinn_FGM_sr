clc; clear; close all;

% Material properties
E = 20e9;       % Young's modulus (Pa)
nu = 0.27;      % Poisson's ratio
rho = 751;      % Density (kg/m³)
H = 0.1;        % Domain size (m)

% Lame parameters
lambda = E*nu/((1+nu)*(1-2*nu));
mu = E/(2*(1+nu));

% Discretization
Nx = 50; Ny = 50;   % Number of elements
dx = H/Nx; dy = H/Ny;
x = 0:dx:H; y = 0:dy:H;
[x_nodes, y_nodes] = meshgrid(x, y);
x_coords = reshape(x_nodes, [], 1);  % Flattened node coordinates
y_coords = reshape(y_nodes, [], 1);
n_nodes = (Nx+1)*(Ny+1);

% Time parameters
dt = 1;      % Time step
T_total = 1; % Total time
Nt = round(T_total/dt);

% Initialize displacement (u_x, u_y at each node)
U = zeros(2*n_nodes, Nt); 
V = zeros(2*n_nodes, 1);   % Velocity
A = zeros(2*n_nodes, 1);   % Acceleration

% Assembly of stiffness (K) and mass (M) matrices
K = sparse(2*n_nodes, 2*n_nodes);
M = sparse(2*n_nodes, 2*n_nodes);

for e = 1:Nx*Ny
    [n1, n2, n3, n4] = getElementNodes(e, Nx, Ny);
    xe = [x_coords(n1) y_coords(n1);
          x_coords(n2) y_coords(n2);
          x_coords(n3) y_coords(n3);
          x_coords(n4) y_coords(n4)];
    [Ke, Me] = elasto2DQuadElement(lambda, mu, rho, xe);
    dofs = [2*n1-1 2*n1 2*n2-1 2*n2 2*n3-1 2*n3 2*n4-1 2*n4];
    K(dofs, dofs) = K(dofs, dofs) + Ke;
    M(dofs, dofs) = M(dofs, dofs) + Me;
end

% Apply initial conditions (sinusoidal displacement)
for n = 1:n_nodes
    xi = x_coords(n); yi = y_coords(n);
    U(2*n-1,1) = sin(pi*xi/H)*sin(pi*yi/H); % u_x
    U(2*n,1) = cos(pi*xi/H)*sin(pi*yi/H);   % u_y
end

% Apply simple boundary conditions (fix bottom edge)
fixed_nodes = find(abs(y_coords) < 1e-10);  % Nodes on y = 0
fixed_dofs = [2*fixed_nodes-1; 2*fixed_nodes];  % u_x and u_y
free_dofs = setdiff(1:2*n_nodes, fixed_dofs);

% Newmark-beta method parameters
beta = 0.25; gamma = 0.5;

for t = 2:Nt
    % Predictor step
    U(:,t) = U(:,t-1) + dt*V + 0.5*dt^2*(1-2*beta)*A;
    
    % Solve for acceleration (on free DOFs only)
    RHS = -K * U(:,t);
    A(free_dofs) = (M(free_dofs, free_dofs) + beta*dt^2*K(free_dofs, free_dofs)) \ RHS(free_dofs);
    
    % Corrector step
    V = V + dt*((1-gamma)*A);        % First half
    U(:,t) = U(:,t) + beta*dt^2*A;   % Displacement correction
    V = V + dt*gamma*A;              % Second half
end

% Reshape displacements into 2D grid
u_x = reshape(U(1:2:end,end), Ny+1, Nx+1);
u_y = reshape(U(2:2:end,end), Ny+1, Nx+1);

figure;

% Plot u_x surface
subplot(1,2,1);
surf(x_nodes, y_nodes, u_x, 'EdgeColor', 'none');
title('u_x at final time');
xlabel('x'); ylabel('y'); zlabel('u_x');
colorbar; axis tight; view(3); shading interp;
colormap jet
grid off;


% Plot u_y surface
subplot(1,2,2);
surf(x_nodes, y_nodes, u_y, 'EdgeColor', 'none');
title('u_y at final time');
xlabel('x'); ylabel('y'); zlabel('u_y');
colorbar; axis tight; view(3); shading interp;
colormap jet
grid off;


%% ========== Helper Functions ==========

function [n1, n2, n3, n4] = getElementNodes(e, Nx, Ny)
    col = mod(e-1, Nx) + 1;
    row = floor((e-1)/Nx) + 1;
    n1 = (row-1)*(Nx+1) + col;
    n2 = n1 + 1;
    n3 = n2 + (Nx+1);
    n4 = n3 - 1;
end

function [Ke, Me] = elasto2DQuadElement(lambda, mu, rho, xe)
    gp = [-1/sqrt(3), -1/sqrt(3); 1/sqrt(3), -1/sqrt(3);
          1/sqrt(3),  1/sqrt(3); -1/sqrt(3), 1/sqrt(3)];
    w = [1, 1, 1, 1];

    Ke = zeros(8,8); Me = zeros(8,8);
    D = [lambda+2*mu lambda 0; lambda lambda+2*mu 0; 0 0 mu];

    for q = 1:4
        xi = gp(q,1); eta = gp(q,2);
        [N, dN_dxi, dN_deta] = shapeFuncQ4(xi, eta);
        J = [dN_dxi; dN_deta] * xe;
        dN = J \ [dN_dxi; dN_deta];
        
        B = zeros(3,8);
        B(1,1:2:end) = dN(1,:);
        B(2,2:2:end) = dN(2,:);
        B(3,1:2:end) = dN(2,:);
        B(3,2:2:end) = dN(1,:);
        
        Ke = Ke + B'*D*B * det(J) * w(q);
        
        Nmat = zeros(2,8);
        Nmat(1,1:2:end) = N;
        Nmat(2,2:2:end) = N;
        Me = Me + (Nmat')*Nmat * rho * det(J) * w(q);
    end
end

function [N, dN_dxi, dN_deta] = shapeFuncQ4(xi, eta)
    N = 0.25 * [(1-xi)*(1-eta), (1+xi)*(1-eta), ...
                (1+xi)*(1+eta), (1-xi)*(1+eta)];
    dN_dxi = 0.25 * [-(1-eta), (1-eta), (1+eta), -(1+eta)];
    dN_deta = 0.25 * [-(1-xi), -(1+xi), (1+xi), (1-xi)];
end
