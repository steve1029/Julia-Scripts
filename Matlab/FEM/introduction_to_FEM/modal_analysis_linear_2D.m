clear; close all; clc
%{
d/dxi(A·sigma(xi)) = rho·A·d^2i/dxi^2
%}

% Parameters
E = 1e7;
A = 1.5;
rho = 2.6e-4;
 
% Node coordinates
p = [0  0 40 40 80 80;
     0 40  0 40  0 40];
number_of_nodes = size(p, 2);
 
% Connectivity
t = [1 3;
     1 4;
     2 4;
     3 4;
     3 5;
     4 5;
     4 6;
     5 6];
number_of_elements = size(t, 1);
 
% c
c = A * E;
 
% Initialization of K and F
K = zeros(2 * number_of_nodes);
M = zeros(2 * number_of_nodes);
F = zeros(2 * number_of_nodes, 1);
 
% Calculation of Ke & assembly of K 
for e = 1 : number_of_elements
   nodes = t(e, :);
   dofs = reshape([2 * nodes - 1; 2 * nodes], 1, 2 * numel(nodes));
   node_coords = p(:, t(e,:));
   n = diff(node_coords, 1, 2);
   n = n / norm(n);
   Q = [n(1) n(2) 0    0;
        0    0    n(1) n(2)];
   local_coords = Q * node_coords(:);
   P = [ones(1, 2); local_coords'];
   length_of_element = abs(det(P));
   C = inv(P);
   grads_phis = C(:, 2);
   Ke = Q' * grads_phis * A * E * grads_phis' * length_of_element * Q;
   
   % Local shape functions
   phi_1 = @(x) C(1, 1) + C(1, 2) * x;
   phi_2 = @(x) C(2, 1) + C(2, 2) * x;
   
   a = local_coords(1);
   b = local_coords(2);
 
   int_Phi = [integral(@(x)phi_1(x) .* phi_1(x), a, b) 0 integral(@(x)phi_1(x) .* phi_2(x), a, b) 0;
              0 integral(@(x)phi_1(x) .* phi_1(x), a, b) 0 integral(@(x)phi_1(x) .* phi_2(x), a, b);
              integral(@(x)phi_1(x) .* phi_2(x), a, b) 0 integral(@(x)phi_2(x) .* phi_2(x), a, b) 0;
              0 integral(@(x)phi_1(x) .* phi_2(x), a, b) 0 integral(@(x)phi_2(x) .* phi_2(x), a, b)];
  
   Me = int_Phi * rho * A;
 
   K(dofs, dofs) = K(dofs, dofs) + Ke;
   M(dofs, dofs) = M(dofs, dofs) + Me;   
end

% Dirichlet boundary
Dirichlet = [1 2];
doffs_Dirichlet = [Dirichlet * 2 - 1, Dirichlet * 2];
K(doffs_Dirichlet, :) = 0;
M(doffs_Dirichlet, :) = 0;
M(doffs_Dirichlet, doffs_Dirichlet) = eye(numel(doffs_Dirichlet));
 
% Sort eigenvalues & eigenvectors
[a, lambda] = eig(M \ K);
[~, permutation] = sort(diag(lambda));
lambda = lambda(permutation, permutation);
a = a(:, permutation);
 
% Angle frequencies
omega = sqrt(lambda);
 
% Frequencies 
frequencies = diag(omega / (2 * pi));
sprintf('f = %.1f Hz\n', frequencies)
 
% Plotting
set(gcf, 'color', 'w')
mode = 5;
gain = 3.5;
t_end = 3 * 2 * pi / omega(mode, mode); % 3 periods
has_been_plot = false;
for time_step = 0 : t_end / 2.5e2 : t_end
    amp = a(:, mode)' * exp(1i * omega(mode, mode) * time_step);
    amp = reshape(amp, 2, number_of_nodes);
    pnew = p + gain * imag(amp);
    if ~has_been_plot
        has_been_plot = true;
        hold on
        for e = 1 : number_of_elements
            nodes = t(e, :);
            h = plot(p(1, nodes), p(2, nodes), 'b--');
            g(e) = plot(pnew(1, nodes), pnew(2, nodes), 'k-o',...
            'MarkerFaceColor', 'k', 'MarkerSize', 10, 'LineWidth', 3);
        end
        axis equal
        axis([-5 85 -5 45])
        axis off
    else
        for e = 1 : number_of_elements
            nodes = t(e, :);
            g(e).XData = pnew(1, nodes);
            g(e).YData = pnew(2, nodes);
        end
    end
    drawnow    
end