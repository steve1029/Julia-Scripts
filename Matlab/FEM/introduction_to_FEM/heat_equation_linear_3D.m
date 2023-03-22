close all; clear; clc
%{
div(k·grad(T)) + s = rho·c·dT/dt 
k·dT/dn|y=0 = q
T(0,08, y, z) = 0
%}
 
% Generate mesh
model = createpde();
model.Geometry = multicuboid(0.02, 0.08, 0.01);
translate(model.Geometry, [0.01 0.04 0]);
pdegplot(model, 'FaceLabels', 'on', 'FaceAlpha', 0.5)
generateMesh(model, 'Hmax', 0.0075, 'GeometricOrder', 'linear');
p = model.Mesh.Nodes;
t = model.Mesh.Elements';
number_of_nodes = size(p, 2);
number_of_elements = size(t, 1);
 
% Parameters
k = 0.1;
s = 6e-3;
q = -1;
 
% Initialization K, M and F
K = zeros(number_of_nodes);
M = zeros(number_of_nodes);
F = zeros(number_of_nodes, 1);
 
% Calculation of Ke, Me en Fe & assembly of K, M and F
for element = 1 : number_of_elements
    nodes = t(element, :); 
    P = [ones(1, 4); p(:, nodes)];
    volume_of_element = abs(det(P)) / 6; 
    C = inv(P);
    grads_phis = C(:, 2:end);
    Ke = grads_phis * k * grads_phis' * volume_of_element;
    mean_of_phis = [1/4; 1/4; 1/4; 1/4];
    Me = mean_of_phis * mean_of_phis' * volume_of_element;
    Fe = mean_of_phis * s * volume_of_element;
    K(nodes, nodes) = K(nodes, nodes) + Ke;
    M(nodes, nodes) = M(nodes, nodes) + Me;
    F(nodes) = F(nodes) + Fe;
end
 
% Neumann boundary
min_y_coordinate = min(p(2, :));
Neumann_elements = [];
for element = 1 : number_of_elements
    nodes = t(element, :);
    node_y_positions = p(2, nodes);
    if (sum(node_y_positions == min_y_coordinate) == 3)
        Neumann_elements = [Neumann_elements, element];
    end
end
 
for element = Neumann_elements
    nodes = t(element, :);
    % Select only the (face) of Neumann boundary
    nodes = nodes(p(2, nodes) == min_y_coordinate);
    P = p(:, nodes);
    area_of_element = 1/2 * norm(cross(P(:, 2) - P(:, 1), P(:, 1) - P(:, 3)));
    Fe = [1/3; 1/3; 1/3] * q * area_of_element ;
    F(nodes) = F(nodes) + Fe;
end
 
% Dirichlet boundary
Dirichlet = find(p(2, :) == max(p(2, :)));
K(Dirichlet, :) = 0;
M(Dirichlet, :) = 0;
M(Dirichlet, Dirichlet) = eye(numel(Dirichlet));
F(Dirichlet) = 0;
 
% Time integration
time = linspace(0, 0.025, 5e2)';
T_0 = zeros(number_of_nodes, 1); 
T_0(Dirichlet) = 1;
dT_dt = @(~, T) M \ (F - K * T);
[~, T] = ode15s(dT_dt, time, T_0);
has_been_plot = false;
for temp = T'
    if ~has_been_plot
        has_been_plot = true;
        h = pdeplot3D(model, 'colormapdata', temp);
        h(2).EdgeColor = 0.75 * [1, 1, 1];
        colormap(jet)
        camzoom(1.10)
        caxis([0 1])
        set(gca, 'Projection', 'perspective')       
        light
        camlight left
    else
        h(2).CData = temp;
    end
    % g = text(p(1, :), p(2, :), p(3, :), num2str(temp, '%.2f'), 'FontSize', 8, 'FontName', 'Consolas', ...
    %    'VerticalAlignment', 'bottom');    
    drawnow
end