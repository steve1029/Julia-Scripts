% 2D Potential flow
close all; clear; clc
 
% Mesh generation
model = createpde();
R1 = [3; 4; 
      0; 3; 3; 0; 
      0; 0; 1; 1];
C1 = [1, 1.0, 0.5, 0.25]';
C1 = [C1; zeros(length(R1) - length(C1), 1)];
gm = [R1, C1];
sf = 'R1-C1';
ns = char('R1', 'C1');
ns = ns';
g = decsg(gm, sf, ns);
geometryFromEdges(model, g);
pdegplot(model, 'EdgeLabels', 'on')
generateMesh(model, 'GeometricOrder', 'linear', 'Hmax', 0.20);
[p, e, t] = meshToPet(model.Mesh);
e = e(1, :);
t = t(1:3, :)';
 
number_of_nodes = size(p, 2);
number_of_elements = size(t, 1);
 
% Initialization of K
K = zeros(number_of_nodes);
F = zeros(number_of_nodes, 1);
 
% Calculation of Ke & assembly of K 
for element = 1 : number_of_elements
    nodes = t(element, :);
    P = [ones(1, 3); p(:, nodes)];
    C = inv(P);
    area_of_element = abs(det(P))/2;
    grads_phis = C(:, 2:3);
    xy_mean = mean(p(:, nodes), 2);
    Ke = grads_phis * grads_phis' * area_of_element;
    K(nodes, nodes) = K(nodes, nodes) + Ke;
end
 
% Neumann boundary
q = -1;
t_Neumann = [];
for element = 1 : number_of_elements
    nodes = t(element, :);
    I = p(1, nodes) == min(p(1, :));
    if( sum(I) == 2)        
        t_Neumann = [t_Neumann; nodes(I)];
    end
end
for element = 1 : size(t_Neumann, 1)
    nodes = t_Neumann(element, :);
    P = p(:, nodes);
    length_of_element = norm(diff(P, 1, 2));
    xy_mean = mean(p(:, nodes), 2);
    Fe = 1/2 * q * length_of_element * [1; 1];
    F(nodes) = F(nodes) + Fe;
end
 
% Dirichlet boundary
Dirichlet = find(p(1, :) == max(p(1, :)));
K(Dirichlet, :) = 0;
K(Dirichlet, Dirichlet) = eye(numel(Dirichlet));
F(Dirichlet) = 0;
 
% Solve
phi = K \ F;
 
% Plotting
trisurf(t, p(1, :), p(2, :), phi, 'edgecolor', ...
    0.5 * [1 1 1], 'facecolor', 'none');
 
% Calculation of velocity vectors at center of elements
Z = [];
V = [];
for element = 1 : number_of_elements
    nodes = t(element, :);
    P = [ones(1, 3); p(:, nodes)];
    C = inv(P);
    grads_phis = C(:, 2:3);
    xy_mean = mean(p(:, nodes), 2);
    Z = [Z; xy_mean'];
    phi_at_nodes = phi(nodes,:);
    V = [V; (grads_phis' * phi_at_nodes)'];
end
 
% Plot velocity field
set(gcf, 'Color', 'white')
view(2)
hold on
quiver(Z(:, 1), Z(:, 2), V(:, 1), V(:, 2), 'k')
axis equal
axis off
 
% Calculate streamlines
x = Z(:, 1); 
y = Z(:, 2); 
u = V(:, 1);
v = V(:, 2);

% Interpolate
U = scatteredInterpolant(x, y, u);
V = scatteredInterpolant(x, y, v);
x_interp = linspace(min(x), max(x), 50);
y_interp = linspace(min(y), max(y), 50);
[X, Y] = meshgrid(x_interp, y_interp);
u_interp = U(X, Y);
v_interp = V(X, Y);

% Startpositions particles
y_ = [linspace(0.1, 0.4, 3), linspace(0.6, 0.9, 3)]; 
x_ = 0.05 * ones(size(y_));  

% Streamline generation
Iverts = [];
for i = 1 : numel(x_)
    h(i) = streamline(stream2(X, Y, u_interp, v_interp, x_(i), y_(i)));
    set(h(i), 'Color', [0.5, 0.5, 1], 'LineWidth', 1.5)   
    verts = stream2(X, Y, u_interp, v_interp, x_(i), y_(i));
    iverts = interpstreamspeed(X, Y, u_interp, v_interp, verts, 0.002);
    Iverts = [Iverts, iverts];
end

% Animation
streamparticles(Iverts, 20, ...
    'Animate', 20, ...
    'ParticleAlignment', 'on', ...
    'Marker', 'o', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', 'b', ...
    'MarkerSize', 5);