// Canal retangular 7 x 1 com um dente na parede de baixo
//
// Canal [-Lc/2, Lc/2] x [-Hc/2, Hc/2], origem no centroide. O dente e' um
// retangulo fino de largura a x altura b, preso na parede de baixo e centrado
// em x = xd (x = 0 e' o meio do canal, a 3,5 da entrada). As laterais do dente
// (faces verticais) tem refino ref_dente vezes maior que o resto da malha.
//
//   gmsh -2 canal_dente.geo -o canal_dente.msh

// ---------------- parametros ----------------
Lc = 7.0;          // comprimento do canal
Hc = 1.0;          // altura do canal
a  = 0.1;          // largura do dente (x)
b  = 0.2;          // altura do dente (y)
xd = 0.0;          // centro do dente em x

lc       = 0.05;           // tamanho no resto da malha
ref_dente = 3.0;           // refino nas laterais do dente
lc_dente = lc/ref_dente;   // tamanho nas laterais do dente

x0 = -Lc/2;  x1 = Lc/2;
y0 = -Hc/2;  y1 = Hc/2;
xe = xd - a/2;  xs = xd + a/2;   // faces do dente (esquerda, direita)
yt = y0 + b;                     // topo do dente

// ---------------- pontos (anti-horario) ----------------
Point(1) = {x0, y0, 0, lc};
Point(2) = {xe, y0, 0, lc};
Point(3) = {xe, yt, 0, lc};
Point(4) = {xs, yt, 0, lc};
Point(5) = {xs, y0, 0, lc};
Point(6) = {x1, y0, 0, lc};
Point(7) = {x1, y1, 0, lc};
Point(8) = {x0, y1, 0, lc};

// ---------------- curvas ----------------
Line(1) = {1, 2};   // parede de baixo, antes do dente
Line(2) = {2, 3};   // lateral esquerda do dente (montante)
Line(3) = {3, 4};   // topo do dente
Line(4) = {4, 5};   // lateral direita do dente (jusante)
Line(5) = {5, 6};   // parede de baixo, depois do dente
Line(6) = {6, 7};   // saida
Line(7) = {7, 8};   // parede de cima
Line(8) = {8, 1};   // entrada

Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
Plane Surface(1) = {1};

// ---------------- grupos fisicos ----------------
Physical Curve("inlet")  = {8};
Physical Curve("outlet") = {6};
Physical Curve("top")    = {7};
Physical Curve("bottom") = {1, 5};
Physical Curve("dente")  = {2, 3, 4};
Physical Surface("surface") = {1};

// ---------------- tamanho de malha ----------------
Field[1] = Distance;
Field[1].CurvesList = {2, 4};    // laterais do dente
Field[1].Sampling = 100;
Field[2] = Threshold;
Field[2].InField  = 1;
Field[2].SizeMin  = lc_dente;
Field[2].SizeMax  = lc;
Field[2].DistMin  = 0.02;
Field[2].DistMax  = 0.3;
Background Field = 2;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeMax = lc;
Mesh.Algorithm = 6;             // Frontal-Delaunay
Mesh.ElementOrder = 1;          // o femns monta MINI/Tri6 a partir de P1
Mesh.MshFileVersion = 2.2;      // mesmo formato das outras malhas
