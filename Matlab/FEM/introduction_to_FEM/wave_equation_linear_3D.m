close all; clear; clc
 
% Generate mesh
model = createpde();
model.Geometry = multicuboid(2, 2, 2);
translate(model.Geometry, [0 0 -1]);
generateMesh(model, 'Hmax', 0.15, 'GeometricOrder', 'linear');
p = model.Mesh.Nodes;
t = model.Mesh.Elements';
number_of_nodes = size(p, 2);
number_of_elements = size(t, 1);

% Parameters
c = 1;
m = 1;
 
% Initialization K and M
K = zeros(number_of_nodes);
M = zeros(number_of_nodes);
 
% Calculation of Ke, Me & assembly of K and F
for element = 1 : number_of_elements
    nodes = t(element, :); 
    P = [ones(1, 4); p(:, nodes)];
    volume_of_element = abs(det(P)) / 6; 
    C = inv(P);
    grads_phis = C(:, 2:end);
    xyz_mean = mean(p(:, nodes), 2);
    Ke = grads_phis * c * grads_phis' * volume_of_element;
    mean_of_phis = [1/4; 1/4; 1/4; 1/4];
    Me = m * (mean_of_phis * mean_of_phis') * volume_of_element;
    K(nodes, nodes) = K(nodes, nodes) + Ke;
    M(nodes, nodes) = M(nodes, nodes) + Me;
end 

% Time integration
time = linspace(0.0, 1.0, 100)';
A = [zeros(number_of_nodes) eye(number_of_nodes); -M \ K zeros(number_of_nodes)];
x = p(1, :);
y = p(2, :);
z = p(3, :);
u_0 = sqrt(x.^2 + y.^2 + z.^2) < 0.15;
du_dt_0 = zeros(1, number_of_nodes);
Z_0 = [u_0, du_dt_0];
dZ_dt = @(~, Z) A * Z;
[~, Z] = ode45(dZ_dt, time, Z_0);

% Slice
Z = Z(:, 1 : number_of_nodes);
stepsize = 0.075;
[x_grid, y_grid, z_grid] = meshgrid(-1:stepsize:1,-1:stepsize:1,-1:stepsize:1);
set(gcf, 'color', 'w')
set(gcf, 'position', [10, 100, 500, 500])
for z_iter = Z'
    dummy_result = createPDEResults(model, z_iter);
    U = interpolateSolution(dummy_result, x_grid, y_grid, z_grid);
    U = reshape(U, size(x_grid));    
    clf
    hold on
    slice(x_grid, y_grid, z_grid, U, 0, 0, 0);
    shading interp
    h = pdeplot3D(model, 'EdgeColor', 0.75 * [1 1 1], 'FaceColor', 'None');
    colormap(1 - jet)
    caxis(0.20 * [0 1])
    set(gca, 'Projection', 'perspective') 
    light
    camlight
    drawnow
end