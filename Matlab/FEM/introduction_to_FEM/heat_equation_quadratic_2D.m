%{
Finite element solution of 2D heat equation (+ animation) using quadratic elements
See also http://www.particleincell.com/blog/2012/matlab-fem
D.E.: d^2T/dx^2 + d^2T/dy^2 + s(x,y) = dT/dt
B.C.: Neumann: dT/dn (Neumann) = -1
B.C.: Dirichlet: corner of omega: 1, rest 0
%}
close all; clear; clc
 
% Mesh generation
load heat_2D.mat
% Neumann boundary
Neumann_e = [
    57 55;
    55 58;
    58 54];

% Dirichlet boundary
Dirichlet_e = [
    1  8;
    8  5;
    5  12;
    12 18;
    4  2;
    2  6;
    6  3;
    3  7;
    7  11;
    11 17;
    17 25;
    25 33;
    33 40;
    40 46;
    46 53;
    53 59;
    59 57;
    54 56;
    56 52;
    52 49;
    49 42;
    42 34;
    34 26];

p_old = p;
t_old = t;
 
% k(x, y), lambda s(x, y), q(x, y)
k = 1;
lambda = 1;
s = @(x, y) 10 * sin(pi * x) .* sin(pi * y);
q = -1;

% Quadratic elements contain six nodes per triangle: add three nodes at the middle of the edges of each triangle
number_of_nodes.old = size(p, 2);
number_of_elements = size(t, 1);
S = zeros(number_of_nodes.old);
counter = number_of_nodes.old + 1;
 
for e = 1:number_of_elements
    nodes = t(e, :); 
    if (S(nodes(1), nodes(2)) == 0)
        S(nodes(1), nodes(2)) = counter;
        S(nodes(2), nodes(1)) = counter;
        p(:, counter) = mean(p(:, [nodes(1) nodes(2)]), 2);
        counter = counter + 1;
    end
    if (S(nodes(2), nodes(3)) == 0)
        S(nodes(2), nodes(3)) = counter;
        S(nodes(3), nodes(2)) = counter;
        p(:, counter) = mean(p(:, [nodes(2) nodes(3)]), 2);
        counter = counter + 1;
    end
    if (S(nodes(1), nodes(3)) == 0)
        S(nodes(1), nodes(3)) = counter;
        S(nodes(3), nodes(1)) = counter;
        p(:, counter) = mean(p(:, [nodes(1) nodes(3)]), 2);
        counter = counter + 1;
    end
    t(e, 4) = S(nodes(1), nodes(2));
    t(e, 5) = S(nodes(2), nodes(3));
    t(e, 6) = S(nodes(1), nodes(3));
end
number_of_nodes.new = size(p, 2);
 
Dirichlet = [];
for i = 1 : size(Dirichlet_e)
    nodes = Dirichlet_e(i, :);
    Dirichlet = [Dirichlet; [nodes(1), S(nodes(1), nodes(2)), nodes(2)]];
end
Neumann = [];
for i = 1 : size(Neumann_e)
    nodes = Neumann_e(i,:);
    Neumann = [Neumann; [nodes(1), S(nodes(1),nodes(2)), nodes(2)]];
end

% Initialisation of K, M and F
K = zeros(number_of_nodes.new);
M = zeros(number_of_nodes.new);
F = zeros(number_of_nodes.new, 1);
 
% Gaussian quadrature points & weights
eta_xi = [2/3 1/6 1/6;
          1/6 2/3 1/6;
          1/6 1/6 2/3];
w = [1/3 1/3 1/3];
 
% Assembly of K, M and F
for e = 1 : number_of_elements
    nodes = t(e, :);
    
    % 6 by 6 matrix with rows: [ones; x; y; x^2; xy; y^2]:
    P = [ones(1, 6); 
         p(:, nodes); 
         p(1, nodes).^2; 
         p(1, nodes) .* p(2, nodes); 
         p(2, nodes).^2];
    area_of_element = abs(det(P(1 : 3, 1 : 3))) / 2;
    
    % Three integration points within each triangle
    ip = eta_xi * p(:, nodes(1 : 3))';
 
    % 6 by 3 matrix with rows: [ones; x; y; x^2; xy; y^2]' (of three integration points)
    IPS = [ones(1, 3); ip'; ip(:, 1)'.^2; ip(:, 1)' .* ip(:, 2)'; ip(:, 2)'.^2];
    dIPS_dx = [zeros(1, 3); ones(1, 3); zeros(1, 3); 2 * ip(:, 1)'; ip(:, 2)'; zeros(1, 3)];
    dIPS_dy = [zeros(1, 3); zeros(1, 3); ones(1, 3); zeros(1, 3); ip(:, 1)'; 2 * ip(:, 2)'];
    IPS_prime = [dIPS_dx(:, 1) dIPS_dy(:, 1) dIPS_dx(:, 2) dIPS_dy(:, 2) dIPS_dx(:, 3) dIPS_dy(:, 3)];    
    
    % Solve for Phi and PhiPrime in integration points
    Phi = P \ IPS; 
    Phi_prime = P \ IPS_prime;
    
    Ke = w(1) * Phi_prime(:, 1:2) * k * Phi_prime(:, 1:2)' * area_of_element + ...
         w(2) * Phi_prime(:, 3:4) * k * Phi_prime(:, 3:4)' * area_of_element + ...
         w(3) * Phi_prime(:, 5:6) * k * Phi_prime(:, 5:6)' * area_of_element;
     
    Me = w(1) * Phi(:, 1) * Phi(:, 1)' * area_of_element + ...
         w(2) * Phi(:, 2) * Phi(:, 2)' * area_of_element + ...
         w(3) * Phi(:, 3) * Phi(:, 3)' * area_of_element;     
 
    Fe = w(1) * Phi(:, 1) * s(ip(1, 1), ip(1, 2)) * area_of_element + ...
         w(2) * Phi(:, 2) * s(ip(2, 1), ip(2, 2)) * area_of_element + ...
         w(3) * Phi(:, 3) * s(ip(3, 1), ip(3, 2)) * area_of_element; 
     
    K(nodes, nodes) = K(nodes, nodes) + Ke;
    M(nodes, nodes) = M(nodes, nodes) + Me;
    F(nodes) = F(nodes) + Fe;    
end
 
% Neumann boundary
xi = 1/2 * [1 + 1/sqrt(3)  1 - 1/sqrt(3);
            1 - 1/sqrt(3)  1 + 1/sqrt(3)];
w = [1/2 1/2];
for e = 1 : size(Neumann, 1)
   nodes = Neumann(e, :);
   % 6 by 3 matrix with rows: [1 x y x^2 xy y^2]' (three nodes on the edge of each element)
   P = [ones(1, 3); p(:, nodes); p(1, nodes).^2; p(1, nodes) .* p(2, nodes); p(2, nodes).^2];
   length_of_element = norm(diff(p(:, nodes([1, 2])), 1, 2));
   
   % Determine the two coordinates of the integration points on the edge of the element belonging 
   % to the Neumann boundary
   ip = xi * p(:, nodes([1, 2]))';
   
   % 6 by 2 matrix with rows: [ones; x; y; x^2; xy; y^2] by the two integration points 
   % on the Neumann element edge
   IPS = [ones(1, 2); ip'; ip(:, 1)'.^2; ip(:, 1)' .* ip(:, 2)'; ip(:, 2)'.^2];
   Phi = P \ IPS;
   
   % Computation of element matrices
   Fe_Gamma = w(1) * Phi(:, 1) * q * length_of_element + ...
              w(2) * Phi(:, 2) * q * length_of_element;
   
   F(nodes) = F(nodes) + Fe_Gamma;
end

% Dirichlet boundary
K(Dirichlet, :) = 0;
M(Dirichlet, :) = 0;
M(:, Dirichlet) = 0;
M(Dirichlet, Dirichlet) = eye(numel(Dirichlet));
F(Dirichlet) = 0;

% Integration
time = linspace(0, 0.25, 2e2)';
T_0 = zeros(size(F));
T_0(Dirichlet(1:4, :)) = 1;
dT_dt = @(~, T) M \ (F - K * T);
[~, T] = ode15s(dT_dt, time, T_0);
 
% Plotting
set(gcf, 'color', 'w')
has_been_plot = false;
for temp = T'
    if ~has_been_plot
        has_been_plot = true;
        h = trisurf(t_old, p_old(1, :), p_old(2, :), temp(1 : number_of_nodes.old), ...
            'EdgeColor', 0.75 * [1 1 1], 'FaceColor', 'interp');
        colormap(jet)
        light
        camlight right
        set(gca, 'Projection', 'perspective')
        caxis([0 1])
        daspect([1 1 2])
        view(70, 20)
        axis([0 1 0 1 min(T(:)) 1])
        axis vis3d
    else
        h.Vertices = [p_old(1, :)', p_old(2, :)', temp(1 : number_of_nodes.old)];
        h.CData = temp(1 : number_of_nodes.old);
    end
    drawnow
end