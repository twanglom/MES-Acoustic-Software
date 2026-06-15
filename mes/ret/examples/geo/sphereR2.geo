// Gmsh project created on Sat Dec 13 22:39:58 2025
SetFactory("OpenCASCADE");
//+
Sphere(1) = {0, 0, 0, 2, -Pi/2, Pi/2, 2*Pi};
//+
Physical Surface("WALL") = {1};
