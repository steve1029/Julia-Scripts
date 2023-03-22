clear; close all; clc
%{ 
non-linear boundary value problem using Picard iteration:
d^2u/dx^2 + u(x)·du/dx - u(x) - exp(2x) = 0
u(0) = 1
du/dx(1) = 1
%}

% Node coordinates
p = linspace(0, 1, 6);
number_of_nodes = numel(p);
 
% Connectivity
t = [1 : number_of_nodes - 1; 2 : number_of_nodes]';
number_of_elements = size(t, 1);
 
% f(x)
f = @(x) -exp(2 * x);
 
% Initial guess    
U = zeros(number_of_nodes, 1);
counter = 0;
U_new = calc_one_iteration(U, t, p, f, number_of_elements, number_of_nodes, counter);
counter = counter + 1;
log_iteration(U_new, U, counter)
 
% Iterate and test residuals
accuracy = 1e-3;
while any(abs(U_new - U) > accuracy)
    U = U_new;
    U_new = calc_one_iteration(U, t, p, f, number_of_elements, number_of_nodes, counter);
    counter = counter + 1;
    log_iteration(U_new, U, counter)
end
 
% Plotting the FEM approximation
plot(p, U, 'o-b', 'Linewidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
 
function[] = log_iteration(U_new, U, counter)
    res = abs(U_new - U);
    U_and_res = [U_new'; res']; 
    U_and_res = reshape(U_and_res(:), numel(U), 2);
    fprintf(['%i:', repmat('\t%8.4f \t%.4f\n', 1, numel(U)), '\n'], counter, U_and_res)
end
 
function [U_new] = calc_one_iteration(U, t, p, f, number_of_elements, number_of_nodes, counter)
    % Calculation of Ke, Me, L, Fe & assembly of K, M, L, F
    % Initialization of K, M, L, F
    persistent K M F
    if ~counter
        K = zeros(number_of_nodes);
        M = zeros(number_of_nodes);
        F = zeros(number_of_nodes, 1);
    end
    L = zeros(number_of_nodes);
    
    for element = 1 : number_of_elements
        nodes = t(element, :);
        P = [ones(1, 2); p(nodes)];
        length_of_element = abs(det(P));
        C = inv(P);
        grads_phis = C(:, 2);
        Phi_mean = [1/2; 1/2];  
        if ~counter
            x_mean = mean(p(nodes));
            Ke = grads_phis * grads_phis' * length_of_element;
            Me = Phi_mean * Phi_mean' * length_of_element;        
            Fe = f(x_mean) * length_of_element * Phi_mean;
            K(nodes, nodes) = K(nodes, nodes) + Ke;
            M(nodes, nodes) = M(nodes, nodes) + Me;
            F(nodes) = F(nodes) + Fe; 
        end
        u_mean = Phi_mean' * U(nodes);
        Le = Phi_mean * u_mean * grads_phis' * length_of_element;
        L(nodes, nodes) = L(nodes, nodes) + Le;  
    end
    
    % Neumann- and Dirichlet boundary
    if ~counter
        F(1) = 1;
        F(end) = F(end) + 1;
    end
    A = K - L + M;
    A(1, :) = 0;
    A(1, 1) = 1; 
    
    % Solving
    U_new = A \ F;
end