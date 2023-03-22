clear; close all; clc
%{
d^2u/dx^2 = 1
Robin boundary condition: du/dx(0) = 5·(u(0) - 0,2)
Dirichlet boundary condition: u(1) = 0
%}

% Node coordinates
p = linspace(0, 1, 11);
number_of_nodes = numel(p);
 
% Connectivity
t = [1 : number_of_nodes - 1; 2 : number_of_nodes]';
number_of_elements = size(t, 1);
 
% Initialization of K and F
K = zeros(number_of_nodes);
F = zeros(number_of_nodes, 1);
 
% Calculation of Ke, Fe & assembly of K and F 
for element = 1 : number_of_elements
   nodes = t(element, :);
   P = [ones(1, 2); p(nodes)];
   length_of_element = abs(det(P));
   C = inv(P);
   grads_phis = C(:, 2);
   Ke = grads_phis * grads_phis' * length_of_element;
   Fe = 1/2 * (-1) * length_of_element * [1; 1];
   K(nodes, nodes) = K(nodes, nodes) + Ke;
   F(nodes) = F(nodes) + Fe;
end
 
% Robin boundary
a = 5;
q = 1;
K(1, 1) = K(1, 1) + a;
F(1) = F(1) + q;
 
% Dirichlet boundary
K(end, :) = 0;
K(end, end) = 1;
F(end) = 0;
 
% Solving
U = K \ F;
 
% Plotting
set(gcf, 'color', 'white')
plot(p, U, 'o-b', 'Linewidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'b')
