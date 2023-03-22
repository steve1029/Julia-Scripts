clear; close all; clc
 
% Set parameters
E = 2.1e11;  % Modulus of elasticity in Pascals
nu = 0.28;   % Poisson's ratio
rho = 7.8e3; % Material density in kg/m^3
 
% Generate mesh
model = createpde('structural', 'modal-solid');
model.Geometry = multicuboid(0.5, 0.05, 0.02);
translate(model.Geometry, [0.5/2 0 0]);
mesh_size = 0.05;
generateMesh(model, 'Hmax', mesh_size, 'GeometricOrder', 'quadratic');
 
% Specify model
structuralProperties(model, 'YoungsModulus', E, 'PoissonsRatio', nu, 'MassDensity', rho);
 
% Define Dirichlet boundary conditions
structuralBC(model, 'Face', 5, 'Constraint', 'fixed');
 
% Modal analysis
frequency_max = 1e4;
results = solve(model, 'FrequencyRange', [0, frequency_max]);
 
mode = 1
magnification_factor = 0.05;
time = linspace(0, 4 * pi, 100);
 
% Plot the geometry and turn on face labels.
subplot(211)
pdegplot(model, 'FaceLabels', 'on');
set(gca, 'Projection', 'perspective')
axis vis3d
 
% Plot mesh
p = model.Mesh.Nodes;
ux = results.ModeShapes.ux;
uy = results.ModeShapes.uy;
uz = results.ModeShapes.uz;
u_three_rows = [1 * ux(:, mode), 1 * uy(:, mode), uz(:, mode)]';
freq_Hz = results.NaturalFrequencies / (2 * pi);
 
subplot(212)
h = pdeplot3D(model, 'ColorMapData', ...
    sign(uz(:, mode)) .* results.ModeShapes.Magnitude(:, mode));
h(2).EdgeColor = 'k';
title(sprintf(['mode = %d\n', 'frequency(Hz): %.1f'], mode, freq_Hz(mode)));
set(gca, 'Projection', 'perspective')
axis vis3d
colormap jet 
ax = axis;
for t = time
    p_new = p + magnification_factor * u_three_rows * sin(t);    
    h(2).Vertices = p_new';
    h(2).CData = uz(:, mode) .* sign(sin(t));
    caxis(max(abs(u_three_rows(:))) * [-1, 1])    
    axis(ax)
    drawnow
end