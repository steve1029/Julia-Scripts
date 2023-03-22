close all; clear; clc
 
% Mesh generation
model = createpde();
l = 0.1;
h = 0.01;
R = [3; 4; 
     0; l; l; 0; 
     0; 0; h; h];
[dl, bt] = decsg(R); 
geometryFromEdges(model, dl);
generateMesh(model, 'GeometricOrder', 'linear', 'Hmax', 0.004);
[p, e, t] = meshToPet(model.Mesh);
edge = e(1, :);
t = t(1:3, :)';
 
number_of_nodes = size(p, 2);
number_of_elements = size(t, 1);
 
% Parameters
E = 70e9;
nu = 0.33;
D = E / (1 - nu^2) * [ 1 nu            0;
                      nu  1            0;
                       0  0 (1 - nu) / 2];
 
% Applied stress
q = [0; -10 / (0.1 * 0.01)];
 
% Initialization of K and F
K = zeros(2 * number_of_nodes);
F = zeros(2 * number_of_nodes, 1);
 
% Calculation of Ke, Fe & assembly of K and F 
for element = 1 : number_of_elements
    nodes = t(element, :);    
    P = [ones(1, 3); p(:, nodes)];
    C = inv(P);
    area_of_element = abs(det(P))/2;
    diff_Phi = C(:, 2:3);
    
    B{element} = [];
    for i = 1 : 3
        b_e = [
                 diff_Phi(i, 1)  0;
                 0               diff_Phi(i, 2);
                 diff_Phi(i, 2)  diff_Phi(i, 1)];
             
        B{element} = [B{element}, b_e];
    end
 
    % Calculate 6 * 6 Ke
    Ke = B{element}' * D * B{element} * area_of_element;
    dofs = reshape([2 * nodes - 1; 2 * nodes], 1, 2 * numel(nodes));        
    K(dofs, dofs) = K(dofs, dofs) + Ke;
end

% Neumann boundary
t_Neumann = [];
for e = 1 : number_of_elements
    nodes = t(e, :);
    I = p(2, nodes) == max(p(2, :));
    if(sum(I) == 2)        
        t_Neumann = [t_Neumann; nodes(I)];
    end
end
 
for element = 1 : size(t_Neumann, 1)
    nodes = t_Neumann(element, :);
    dofs_Neumann = reshape([2 * nodes - 1; 2 * nodes], 1, 2 * numel(nodes));
    P = p(:, nodes);
    length_of_element = norm(diff(P, 1, 2));
    H_mean = 1/2 * repmat(eye(2), 1, 2);
    Fe = H_mean' * q * length_of_element;
    F(dofs_Neumann) = F(dofs_Neumann) + Fe;
end
 
% Dirichlet boundary
Dirichlet = edge(p(1, edge) == 0);
dofs_Dirichlet = [2 * Dirichlet - 1, 2 * Dirichlet];
K(dofs_Dirichlet, :) = 0;
K(dofs_Dirichlet, dofs_Dirichlet) = eye(numel(dofs_Dirichlet));
F(dofs_Dirichlet) = 0;
 
% Solve
U = K \ F;
min(U(:))
displacements = [U(1 : 2 : end), U(2 : 2 : end)]';
magnification = 5e2;
p_new = p + magnification * displacements;
 
% Calculate stresses
S = zeros(1 * number_of_nodes, 3);
node_occurences = zeros(1 * number_of_nodes, 1);
for element = 1 : number_of_elements
    nodes = t(element, :);
    displacement_nodes_of_element = displacements(:, nodes);
    U_e = displacement_nodes_of_element(:);
    s = D * B{element} * U_e;
    S(nodes', :) = S(nodes', :) + repmat(s', 3, 1);
end
 
% Determine node frequencies (occurrences)
for i = 1 : number_of_nodes
    node_occurences(i) = numel(find(t == i));
end
 
S = S ./ repmat(node_occurences, 1, 3);

% Plotting
STRESS_COMPONENT = 1;
splot = S(:, STRESS_COMPONENT);
 
set(gcf, 'color', 'w')
colormap jet
hold on
trisurf(t, p_new(1, :), p_new(2, :), zeros(1, number_of_nodes), splot, 'EdgeColor', 'k', 'FaceColor', 'interp');
trisurf(t, p(1, :), p(2, :), zeros(1, number_of_nodes), 'EdgeColor',  0.75 * [1 1 1], 'FaceColor', 'none');
colorbar
view(2)
axis equal
axis tight