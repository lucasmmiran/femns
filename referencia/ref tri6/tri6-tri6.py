def convertGmshTri6ToTri6(_self):
 """Realiza a operacao associada a convert gmsh tri6 to tri6.
 Usa os dados geometricos, topologicos ou de campo ja armazenados no objeto.
 Atualiza resultados internos ou devolve quantidades auxiliares para o solver.
 """
 # ----- INICIO: conversao TRI6 (Gmsh) --> TRI6 (Navier-Stokes) ------------- #
 # Set tri element to X,Y and IEN arrays. The tri element consists in
 # the same X,Y,IEN struct with additional edge nodes. Therefore X,Y and
 # IEN should be resized to accomadate the 3 more coordinates.
 #
 #     2D QUAD-ELEMENT TRI6
 # (zienkiewicz vol1 pag 180 - 182)
 #
 #       Quad triangle:
 #
 #             v3
 #             o
 #            / \
 #        v6 o   o v5
 #          /     \
 #         o - o - o
 #       v1    v4   v2
 #
 # numeracao:
 #  1) vertices dos triangulos
 #  2) nos das arestas (edge)
 # cria lista de vertices e arestas (malha triangulo)
 start = time.time()

 _self.vertlist = np.unique( _self.IEN[:,0:3].flatten() ).tolist()
 _self.edgelist = np.unique( _self.IEN[:,3:6].flatten() ).tolist()
 _self.numVerts = len(_self.vertlist)
 _self.numEdges = len(_self.edgelist)
 _self.convertp = -1*np.ones(( max(_self.vertlist)+1 ),dtype='int')
 _self.converte = -1*np.ones(( max(_self.edgelist)+1 ),dtype='int')

 # vetores coordenadas X,Y
 X = np.zeros( (_self.numNodes),dtype='float' )
 Y = np.zeros( (_self.numNodes),dtype='float' )
 count = 0
 for v in _self.vertlist:
  _self.convertp[v] = count
  X[count] = _self.X[v]
  Y[count] = _self.Y[v]
  count += 1

 for e in _self.edgelist:
  _self.converte[e] = count
  X[count] = _self.X[e]
  Y[count] = _self.Y[e]
  count += 1

 # copy to class
 _self.X = X.copy()
 _self.Y = Y.copy()
 _self.Xinit = _self.X.copy() # copy initial point distribution
 _self.Yinit = _self.Y.copy()

 # matriz de conectividade IEN
 IEN = np.zeros( (_self.numElems,6),dtype='int' )
 for i in range(0,_self.numElems):
  for j in range(0,3):
   IEN[i,j]   = _self.convertp[ _self.IEN[i,j] ]   # j=0,1,2 (vertex)
   IEN[i,j+3] = _self.converte[ _self.IEN[i,j+3] ] # j=3,4,5 (edge node)

 # copy to class
 _self.IEN = IEN.copy()

 # cria lista de vertices e arestas (malha contorno)
 # numeracao:
 #  1) vertices dos segmentos de reta
 #  2) nos das arestas (edge)
 _self.bvertlist = np.unique( _self.IENbound[:,0:2].flatten() ).tolist()
 _self.bedgelist = np.unique( _self.IENbound[:,2:3].flatten() ).tolist()
 _self.numVertsb = len(_self.bvertlist)
 _self.numEdgesb = len(_self.bedgelist)
 _self.numNodesb = _self.numVertsb+_self.numEdgesb

 # matriz de conectividade de contorno IENbound
 # o ---- o ---- o   --->  o ---- o ---- o
 # v1    v2     ve        v1 ve+numVerts v2
 IENbound = np.zeros( (_self.numElemsb,3),dtype='int' )
 for i in range(0,_self.numElemsb):
  IENbound[i,0] = _self.convertp[ _self.IENbound[i,0] ] # vertex 1 (ind FEM global)
  IENbound[i,2] = _self.convertp[ _self.IENbound[i,1] ] # vertex 2 (ind FEM global)
  IENbound[i,1] = _self.converte[ _self.IENbound[i,2] ] # edge node (ind FEM global)

 # copy to class
 _self.IENbound = IENbound.copy()

 _self.element = 'P2P1'
 end = time.time()
 _self.time = str(round(end-start,3))
 # -----------------------------------------------------------------------------
 # ----- FIM: conversao TRI6 (Gmsh) --> TRI6 (Navier-Stokes) ---------------- #
