clear; close all; clc
% Set parameters of pde: -div(c*grad(u)) + a*u = f
 
% Generate mesh
model = createpde('structural', 'static-solid');
model.Geometry = multicuboid(0.1, 0.01, 0.01);
translate(model.Geometry, [0.1/2 0 0]);
generateMesh(model, 'Hmax', 0.5e-2, 'GeometricOrder', 'quadratic');
structuralProperties(model, 'YoungsModulus', 2e11, 'PoissonsRatio', 0.3);
 
% Apply a Neumann Boundary Conditions: n·(c*u) + qu = g
% Applied stress in Pascals
applied_stress = 2e7;
q = @(region, state) [0; 0; -applied_stress * (1 - region.x / 0.1)];
structuralBoundaryLoad(model, 'Face', 2, 'SurfaceTraction', q);
 
% Define Dirichlet Boundary Conditions
structuralBC(model, 'Face', 5, 'Constraint', 'fixed');
 
% Calculate the Solution of -div(c*grad(u)) + a*u = f
result = solve(model);
 
% Plot the geometry and turn on face labels.
subplot(211)
pdegplot(model, 'FaceLabels', 'on')
set(gca, 'Projection', 'perspective')
axis vis3d
camlight left
light

% Plot deformed mesh and sxx component
subplot(212)
h = pdeplot3D(model);
axis vis3d
set(gca, 'Projection', 'perspective')
h.EdgeColor = 'k';
h.FaceColor = 'None';
hold on
delete(findobj(gca, 'type', 'Quiver'));
g = pdeplot3D(model, 'ColormapData', result.Stress.sxx, ...
    'Deformation', result.Displacement, 'DeformationScaleFactor', 3);
g(2).EdgeColor = 'k';
g(2).Parent.Parent.Position = h.Parent.Position;
min_z_displacement = 1000 * min(result.Displacement.uz);
format_spec = "min displacement = %.2f [mm]";
str = sprintf(format_spec, min_z_displacement);
title(str)
colorbar off
 
% Analytical
l = 0.1;
b = 0.01;
h = 0.01;
I = b * h^3 / 12;
w = -applied_stress * l * b / l;
analytical_deflection_in_mm = ...
    1000 * w * l^4 / (30 * model.MaterialProperties.MaterialAssignments.YoungsModulus * I)