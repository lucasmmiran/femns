// Valvula esfera 2D -- malha de partida (theta0 = 10 graus)
//
// Canal [-Lp, Lp] x [-H, H]; a esfera de raio R = sqrt(H^2 + (L1/2)^2) tem
// um furo de largura 2H e comprimento L1, girado de theta (anti-horario).
// Nas 4 juncoes tubo/esfera a parede segue o proprio circulo da esfera
// (mesma geometria de referencia/malha movel/estrategia_pontos_moveis.md):
//
//   cima : tubo -> arco phi2->phi2+theta -> corda -> arco phi1+theta->phi1 -> tubo
//   baixo: simetrico por reflexao no ponto (p -> -p)
//
// Cada uma das 8 quinas (tubo/arco e arco/corda) e' trocada por um filete
// de raio rf, calculado aqui mesmo: o centro fica a rf da reta e a R +- rf
// do centro da esfera. Tudo parametrico -- mude theta/rf/tamanhos e regere:
//
//   gmsh -2 valvula_10g.geo -o valvula_10g.msh

// ---------------- parametros ----------------
H     = 0.5;            // meia-altura do canal
L1    = 2.0;            // comprimento da parede do furo (corda)
Lp    = 5.0;            // meio-comprimento do canal
theta = 10*Pi/180;      // rotacao inicial da esfera
rf    = 0.03;           // raio dos filetes

lc_far    = 0.06;       // tamanho longe da valvula
lc_valve  = 0.02;       // tamanho na corda e nos arcos
lc_fillet = rf/3;       // tamanho nos filetes

// ---------------- geometria derivada ----------------
R  = Sqrt(H^2 + (L1/2)^2);
ct = Cos(theta);  st = Sin(theta);
// d = (ct, st) direcao da corda; n = (-st, ct) normal (para cima)

If (theta >= 2*Asin(H/R))
  Error("theta >= angulo de fechamento 2*asen(H/R)");
EndIf
If (rf >= (R - H)/2)
  Error("rf grande demais");
EndIf

// Filete tubo/esfera, junta de cima-esquerda (fluido convexo):
// centro a rf abaixo de y=H e a R+rf de O.
c1y = H - rf;  c1x = -Sqrt((R + rf)^2 - c1y^2);
// Filete esfera/corda, cima-esquerda (fluido reentrante: quina da esfera):
// centro dentro da esfera (R-rf) e a rf acima da corda.
t2  = Sqrt((R - rf)^2 - (H + rf)^2);
c2x = -(H + rf)*st - t2*ct;  c2y = (H + rf)*ct - t2*st;
// Filete corda/cavidade, cima-direita (fluido convexo):
// centro dentro da esfera (R-rf) e a rf abaixo da corda.
t3  = Sqrt((R - rf)^2 - (H - rf)^2);
c3x = -(H - rf)*st + t3*ct;  c3y = (H - rf)*ct + t3*st;
// Filete cavidade/tubo, cima-direita (fluido reentrante: quina da carcaca):
// centro a rf acima de y=H e a R+rf de O.
c4y = H + rf;  c4x = Sqrt((R + rf)^2 - c4y^2);

// Pontos de tangencia
a1x = c1x;                 a1y = H;                  // tubo
a2x = c1x*R/(R + rf);      a2y = c1y*R/(R + rf);     // esfera
a3x = c2x*R/(R - rf);      a3y = c2y*R/(R - rf);     // esfera
a4x = c2x + rf*st;         a4y = c2y - rf*ct;        // corda
b4x = c3x - rf*st;         b4y = c3y + rf*ct;        // corda
b3x = c3x*R/(R - rf);      b3y = c3y*R/(R - rf);     // cavidade
b2x = c4x*R/(R + rf);      b2y = c4y*R/(R + rf);     // cavidade
b1x = c4x;                 b1y = H;                  // tubo

// Os dois filetes de cada arco nao podem se cruzar
If (Atan2(a3y, a3x) <= Atan2(a2y, a2x) || Atan2(b3y, b3x) <= Atan2(b2y, b2x))
  Error("filetes se sobrepoem no arco: diminua rf ou aumente theta");
EndIf

// ---------------- pontos ----------------
Point(1)  = {0, 0, 0, lc_far};            // centro da esfera

// cima (esquerda -> direita)
Point(10) = {-Lp, H, 0, lc_far};
Point(11) = {a1x, a1y, 0, lc_far};
Point(12) = {c1x, c1y, 0, lc_far};        // centro filete
Point(13) = {a2x, a2y, 0, lc_far};
Point(14) = {a3x, a3y, 0, lc_far};
Point(15) = {c2x, c2y, 0, lc_far};        // centro filete
Point(16) = {a4x, a4y, 0, lc_far};
Point(17) = {b4x, b4y, 0, lc_far};
Point(18) = {c3x, c3y, 0, lc_far};        // centro filete
Point(19) = {b3x, b3y, 0, lc_far};
Point(20) = {b2x, b2y, 0, lc_far};
Point(21) = {c4x, c4y, 0, lc_far};        // centro filete
Point(22) = {b1x, b1y, 0, lc_far};
Point(23) = {Lp, H, 0, lc_far};

// baixo = reflexao no ponto dos de cima (p -> -p)
For i In {10:23}
  xyz[] = Point{i};
  Point(i + 20) = {-xyz[0], -xyz[1], 0, lc_far};
EndFor
// 30 = (Lp,-H) ... 43 = (-Lp,-H)

// ---------------- curvas ----------------
// cima, esquerda -> direita
Line(1)    = {10, 11};          // tubo de entrada
Circle(2)  = {11, 12, 13};      // filete
Circle(3)  = {13, 1, 14};       // arco da esfera (bloqueia a entrada)
Circle(4)  = {14, 15, 16};      // filete
Line(5)    = {16, 17};          // corda de cima
Circle(6)  = {17, 18, 19};      // filete
Circle(7)  = {19, 1, 20};       // arco da cavidade (carcaca)
Circle(8)  = {20, 21, 22};      // filete
Line(9)    = {22, 23};          // tubo de saida
// baixo, direita -> esquerda (imagem das curvas de cima)
Line(21)   = {30, 31};
Circle(22) = {31, 32, 33};
Circle(23) = {33, 1, 34};       // arco da esfera (bloqueia a saida)
Circle(24) = {34, 35, 36};
Line(25)   = {36, 37};          // corda de baixo
Circle(26) = {37, 38, 39};
Circle(27) = {39, 1, 40};       // arco da cavidade (carcaca)
Circle(28) = {40, 41, 42};
Line(29)   = {42, 43};
// entrada e saida
Line(40) = {43, 10};
Line(41) = {23, 30};

// fronteira no sentido anti-horario: baixo (esq->dir), saida, cima (dir->esq), entrada
Curve Loop(1) = {-29, -28, -27, -26, -25, -24, -23, -22, -21,
                 -41, -9, -8, -7, -6, -5, -4, -3, -2, -1, -40};
Plane Surface(1) = {1};

// ---------------- grupos fisicos (mesmos nomes das malhas valve_*) ------------
Physical Curve("inlet")       = {40};
Physical Curve("outlet")      = {41};
Physical Curve("bottom")      = {21, 29};
Physical Curve("upper")       = {1, 9};
Physical Curve("valve_up")    = {5};
Physical Curve("valve_down")  = {25};
Physical Curve("outter_wall") = {2, 3, 4, 22, 23, 24};   // superficie da esfera + filetes
Physical Curve("inner_wall")  = {6, 7, 8, 26, 27, 28};   // cavidade da carcaca + filetes
Physical Surface("surface")   = {1};

// ---------------- tamanho de malha ----------------
Field[1] = Distance;
Field[1].CurvesList = {3, 5, 7, 23, 25, 27};
Field[1].Sampling = 200;
Field[2] = Threshold;
Field[2].InField  = 1;
Field[2].SizeMin  = lc_valve;
Field[2].SizeMax  = lc_far;
Field[2].DistMin  = 0.05;
Field[2].DistMax  = 1.0;

Field[3] = Distance;
Field[3].CurvesList = {2, 4, 6, 8, 22, 24, 26, 28};
Field[3].Sampling = 50;
Field[4] = Threshold;
Field[4].InField  = 3;
Field[4].SizeMin  = lc_fillet;
Field[4].SizeMax  = lc_far;
Field[4].DistMin  = 0.01;
Field[4].DistMax  = 0.3;

Field[5] = Min;
Field[5].FieldsList = {2, 4};
Background Field = 5;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;             // Frontal-Delaunay
Mesh.ElementOrder = 1;          // o femns monta MINI/Tri6 a partir de P1
Mesh.MshFileVersion = 2.2;      // mesmo formato das outras malhas
