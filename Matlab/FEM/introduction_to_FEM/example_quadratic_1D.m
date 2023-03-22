clear; close all; clc
%{
d^2u/dx^2 + 4·pi^2·u(x) = 0
u(0) = 0
du/dx(1) = 1
%}

% Node coordinates
n = 10;
p = linspace(0, 1, n);
number_of_nodes.old = numel(p);
 
% Connectivity
t = [1 : number_of_nodes.old  - 1; 2 : number_of_nodes.old]';
number_of_elements = size(t, 1);
 
% Add extra node in center of eache element
S = zeros(number_of_nodes.old);
counter = number_of_nodes.old + 1;
for e = 1 : number_of_elements
    nodes = t(e, :);
    p(:, counter) = mean(p(nodes(1 : 2 )));
    t(e, 3) = counter;
    counter = counter + 1;  
end
number_of_nodes.new = numel(p);
 
% c(x), lambda, f(x)
c = 1;
lambda = 4 * pi^2;
 
% Initialization of K, M and F
K = zeros(number_of_nodes.new);
M = zeros(number_of_nodes.new);
F = zeros(number_of_nodes.new, 1);
 
% Calculation of Ke, Fe & assembly of K and F
xi = 1/2 * [1 + 1/sqrt(3)  1 - 1/sqrt(3);
            1 - 1/sqrt(3)  1 + 1/sqrt(3)];
w = [1/2; 1/2];
 
hold on
for e = 1 : number_of_elements
   nodes = t(e, :);
   node_coords = p(nodes);
   plot(node_coords([1 2]), [0, 0], 'k-o', 'MarkerSize', 4, 'MarkerFaceColor', 'k');
   P = [ones(1, 3); node_coords; node_coords.^2];
   length_of_element = abs(det(P([1 2], [1 2])));
   
   % Integration points
   ip = xi * node_coords([1 2])';
   plot(ip, [0, 0], 'ko', 'MarkerSize', 3, 'MarkerFaceColor', 'y');
   
   % Calculate value of phi's at integration points
   IPS = [ones(1, 2); ip'; ip.^2'];   
   Phi = P \ IPS;
   
   % Calculate value of derivatives of phi's at integration points
   IPS_prime = [0 0; 1 1; 2 * ip'];
   Phi_prime = P \ IPS_prime;
   
   Ke = w(1) * Phi_prime(:, 1) * c * Phi_prime(:, 1)' * length_of_element + ...
        w(2) * Phi_prime(:, 2) * c * Phi_prime(:, 2)' * length_of_element;
    
   Me = w(1) * Phi(:, 1) * Phi(:, 1)' * length_of_element +...
        w(2) * Phi(:, 2) * Phi(:, 2)' * length_of_element;
      
   K(nodes, nodes) = K(nodes, nodes) + Ke;
   M(nodes, nodes) = M(nodes, nodes) + Me;
end
 
% Neumann boundary
q = c * 1;    
F(n) = F(n) + q; 
 
% Dirichlet boundary
A = (K - lambda * M);
A(1, :) = 0;
A(1, 1) = 1;
F(1) = 0;
 
% Solving
U = A \ F;
 
% Plotting
set(gcf, 'color', 'w')
plot(p(1 : number_of_nodes.old), U(1 : number_of_nodes.old), ... 
    'o-b', 'Linewidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'b');
fplot(@(x) 1 / (2 * pi) * sin(2 * pi * x), [0, 1], 'k')
grid on