# -----------------------------------------------------------------------------
# ASSEMBLY
# loop para construcao das matrizes MEF
start = time.time()

iv = np.zeros( (numElems*6*6),dtype='int' )
jv = np.zeros( (numElems*6*6),dtype='int' )
ip = np.zeros( (numElems*6*3),dtype='int' )
jp = np.zeros( (numElems*6*3),dtype='int' )
ipd = np.zeros( (numElems*3*6),dtype='int' )
jpd = np.zeros( (numElems*3*6),dtype='int' )
k_data = np.zeros( (numElems*6*6),dtype='float' )
m_data = np.zeros( (numElems*6*6),dtype='float' )
#cx_data = np.zeros( (numElems*6*6),dtype='float' )
#cy_data = np.zeros( (numElems*6*6),dtype='float' )
gx_data = np.zeros( (numElems*6*3),dtype='float' )
gy_data = np.zeros( (numElems*6*3),dtype='float' )
dx_data = np.zeros( (numElems*3*6),dtype='float' )
dy_data = np.zeros( (numElems*3*6),dtype='float' )

# # Gaussian Quadrature
# quad = Elements.Quad(X,Y)

for e in range(0,numElems):
 v = IEN[e]

 area = (1/2)*( X[v[2]]*( Y[v[0]]-Y[v[1]]) \
              + X[v[0]]*( Y[v[1]]-Y[v[2]]) \
              + X[v[1]]*(-Y[v[0]]+Y[v[2]]) )

 bi = Y[v[1]]-Y[v[2]]
 bj = Y[v[2]]-Y[v[0]]
 bk = Y[v[0]]-Y[v[1]]

 ci = X[v[2]]-X[v[1]]
 cj = X[v[0]]-X[v[2]]
 ck = X[v[1]]-X[v[0]]

 mele = (area/180)*np.array([[ 6.0,-1.0,-1.0, 0.0,-4.0, 0.0],
                             [-1.0, 6.0,-1.0, 0.0, 0.0,-4.0],
                             [-1.0,-1.0, 6.0,-4.0, 0.0, 0.0],
                             [ 0.0, 0.0,-4.0,32.0,16.0,16.0],
                             [-4.0, 0.0, 0.0,16.0,32.0,16.0],
                             [ 0.0,-4.0, 0.0,16.0,16.0,32.0]])

 kxele = (1.0/(12.0*area))*np.array([
    [ 3*bi**2, -bi*bj, -bi*bk, 4*bi*bj,0.0,     4*bi*bk ],
    [ -bi*bj, 3*bj**2,-bj*bk,4*bi*bj, 4*bj*bk, 0.0 ],
    [ -bi*bk, -bj*bk, 3*bk**2,0.0,    4*bj*bk, 4*bi*bk ],
    [ 4*bi*bj, 4*bi*bj, 0.0, 8*(bi**2+bi*bj+bj**2),
                             4*(bj**2+bj*bk+bi*bj+2*bi*bk),
                             4*(bi**2 + bi*bk + bi*bj + 2*bj*bk) ],
    [ 0.0, 4*bj*bk, 4*bj*bk, 4*(bi*bj + 2*bi*bk + bj**2 + bj*bk),
                             8*(bj**2 + bj*bk + bk**2),
                             4*(2*bi*bj + bi*bk + bj*bk + bk**2) ],
    [ 4*bi*bk, 0.0, 4*bi*bk, 4*(bi**2 + bi*bj + bi*bk + 2*bj*bk),
                             4*(2*bi*bj + bi*bk + bj*bk + bk**2),
                             8*(bi**2 + bi*bk + bk**2) ] ])

 kyele = (1.0/(12.0*area))*np.array([
    [ 3*ci**2, -ci*cj, -ci*ck, 4*ci*cj,0.0,     4*ci*ck ],
    [ -ci*cj, 3*cj**2,-cj*ck,4*ci*cj, 4*cj*ck, 0.0 ],
    [ -ci*ck, -cj*ck, 3*ck**2,0.0,    4*cj*ck, 4*ci*ck ],
    [ 4*ci*cj, 4*ci*cj, 0.0, 8*(ci**2+ci*cj+cj**2),
                             4*(cj**2+cj*ck+ci*cj+2*ci*ck),
                             4*(ci**2 + ci*ck + ci*cj + 2*cj*ck) ],
    [ 0.0, 4*cj*ck, 4*cj*ck, 4*(ci*cj + 2*ci*ck + cj**2 + cj*ck),
                             8*(cj**2 + cj*ck + ck**2),
                             4*(2*ci*cj + ci*ck + cj*ck + ck**2) ],
    [ 4*ci*ck, 0.0, 4*ci*ck, 4*(ci**2 + ci*cj + ci*ck + 2*cj*ck),
                             4*(2*ci*cj + ci*ck + cj*ck + ck**2),
                             8*(ci**2 + ci*ck + ck**2) ] ])

 kxyele = (1.0/(12.0*area))*np.array([
    [ 3*bi*ci, -bi*cj, -bi*ck, 4*bi*cj, 0.0, 4*bi*ck ],
    [ -bj*ci, 3*bj*cj, -bj*ck, 4*bj*ci, 4*bj*ck, 0.0 ],
    [ -bk*ci, -bk*cj, 3*bk*ck, 0.0, 4*bk*cj, 4*bk*ci ],
    [  4*bj*ci, 4*bi*cj, 0.0, 4*(2*bi*ci + bi*cj + bj*ci + 2*bj*cj),
                              4*(bi*cj + 2*bi*ck + bj*cj + bj*ck),
                              4*(bi*ci + bi*ck + bj*ci + 2*bj*ck) ],
    [  0.0, 4*bk*cj, 4*bj*ck, 4*(bj*ci + bj*cj + 2*bk*ci + bk*cj),
                              4*(2*bj*cj + bj*ck + bk*cj + 2*bk*ck),
                              4*(2*bj*ci + bj*ck + bk*ci + bk*ck) ],
    [ 4*bk*ci, 0.0, 4*bi*ck,  4*(bi*ci + bi*cj + bk*ci + 2*bk*cj),
                              4*(2*bi*cj + bi*ck + bk*cj + bk*ck),
                              4*(2*bi*ci + bi*ck + bk*ci + 2*bk*ck) ] ])

 gxslipele = (1.0/6.0) * np.array([ [ 0.0, 0.0, 0.0],
                                    [ 0.0, 0.0, 0.0],
                                    [ 0.0, 0.0, 0.0],
                                    [  bi,  bj,  bk],
                                    [  bi,  bj,  bk],
                                    [  bi,  bj,  bk] ])

 gyslipele = (1.0/6.0) * np.array([ [ 0.0, 0.0, 0.0],
                                    [ 0.0, 0.0, 0.0],
                                    [ 0.0, 0.0, 0.0],
                                    [  ci,  cj,  ck],
                                    [  ci,  cj,  ck],
                                    [  ci,  cj,  ck] ])

 gxele = (1.0/6.0) * np.array([ [       bi,        0.0,       0.0],
                                [       0.0,        bj,       0.0],
                                [       0.0,       0.0,        bk],
                                [ bi + 2*bj, 2*bi + bj,   bi + bj],
                                [ bj + bk,   bj + 2*bk, 2*bj + bk],
                                [ bi + 2*bk,   bi + bk, 2*bi + bk] ])

 gyele = (1.0/6.0) * np.array([ [       ci,        0.0,       0.0],
                                [       0.0,        cj,       0.0],
                                [       0.0,       0.0,        ck],
                                [ ci + 2*cj, 2*ci + cj,   ci + cj],
                                [ cj + ck,   cj + 2*ck, 2*cj + ck],
                                [ ci + 2*ck,   ci + ck, 2*ci + ck] ])

 dxele = gxele.T
 dyele = gyele.T

 gvxele = (1.0/30.0) * np.array([
                       [ 2*bi, -bj, -bk, -bi+2*bj, -(bj+bk), -bi+2*bk ],
                       [  -bi, 2*bj, -bk, 2*bi-bj, -bj+2*bk, -(bi+bk) ],
                       [ -bi, -bj, 2*bk, -(bi+bj), 2*bj-bk, 2*bi-bk ],
                       [ 3*bi, 3*bj, -bk, 8*(bi+bj), 4*(bj+2*bk), 4*(bi+2*bk) ],
                       [ -bi, 3*bj, 3*bk, 4*(2*bi+bj), 8*(bj+bk), 4*(2*bi+bk) ],
                       [ 3*bi, -bj, 3*bk, 4*(bi+2*bj), 4*(2*bj+bk), 8*(bi+bk) ] ])

 gvyele = (1.0/30.0) * np.array([
                       [ 2*ci, -cj, -ck, -ci+2*cj, -(cj+ck), -ci+2*ck ],
                       [  -ci, 2*cj, -ck, 2*ci-cj, -cj+2*ck, -(ci+ck) ],
                       [ -ci, -cj, 2*ck, -(ci+cj), 2*cj-ck, 2*ci-ck ],
                       [ 3*ci, 3*cj, -ck, 8*(ci+cj), 4*(cj+2*ck), 4*(ci+2*ck) ],
                       [ -ci, 3*cj, 3*ck, 4*(2*ci+cj), 8*(cj+ck), 4*(2*ci+ck) ],
                       [ 3*ci, -cj, 3*ck, 4*(ci+2*cj), 4*(2*cj+ck), 8*(ci+ck) ] ])


 [v1,v2,v3,v4,v5,v6] = IEN[e]

 # # definicao das matrizes do elemento por Quadratura Guassiana
 # quad.getMSlip( IEN[e] )
 # #quad.getMAnalytic( IEN[e] )

 iv[e*6*6:(e+1)*6*6] = 6*[v1] + \
                       6*[v2] + \
                       6*[v3] + \
                       6*[v4] + \
                       6*[v5] + \
                       6*[v6]
 jv[e*6*6:(e+1)*6*6] = 6*[v1,v2,v3,v4,v5,v6]
 ip[e*6*3:(e+1)*6*3] = 3*[v1] + \
                       3*[v2] + \
                       3*[v3] + \
                       3*[v4] + \
                       3*[v5] + \
                       3*[v6]
 jp[e*6*3:(e+1)*6*3] = 6*[v1,v2,v3]

 ipd[e*3*6:(e+1)*3*6] = 6*[v1] + \
                        6*[v2] + \
                        6*[v3]
 jpd[e*3*6:(e+1)*3*6] = 3*[v1,v2,v3,v4,v5,v6]

 # m_data[e*6*6:(e+1)*6*6] = quad.mass.flatten()
 # k_data[e*6*6:(e+1)*6*6] = quad.kxx.flatten() + \
 #                           quad.kyy.flatten()
 # #cx_data[e*6*6:(e+1)*6*6] = quad.gvx.flatten()
 # #cy_data[e*6*6:(e+1)*6*6] = quad.gvy.flatten()
 # gx_data[e*6*3:(e+1)*6*3] = quad.gx.flatten()
 # gy_data[e*6*3:(e+1)*6*3] = quad.gy.flatten()
 # dx_data[e*3*6:(e+1)*3*6] = quad.dx.flatten()
 # dy_data[e*3*6:(e+1)*3*6] = quad.dy.flatten()
 m_data[e*6*6:(e+1)*6*6] = mele.flatten()
 k_data[e*6*6:(e+1)*6*6] = kxele.flatten() + \
                           kyele.flatten()
 #cx_data[e*6*6:(e+1)*6*6] = quad.gvx.flatten()
 #cy_data[e*6*6:(e+1)*6*6] = quad.gvy.flatten()
 gx_data[e*6*3:(e+1)*6*3] = gxslipele.flatten()
 gy_data[e*6*3:(e+1)*6*3] = gyslipele.flatten()
 dx_data[e*3*6:(e+1)*3*6] = dxele.flatten()
 dy_data[e*3*6:(e+1)*3*6] = dyele.flatten()

M =  coo_matrix((m_data, (iv,jv)),shape=(numNodes,numNodes))
K =  coo_matrix((k_data, (iv,jv)),shape=(numNodes,numNodes))
#Cx = coo_matrix((cx_data,(iv,jv)),shape=(numNodes,numNodes))
#Cy = coo_matrix((cy_data,(iv,jv)),shape=(numNodes,numNodes))
Gx = coo_matrix((gx_data,(ip,jp)),shape=(numNodes,numVerts))
Gy = coo_matrix((gy_data,(ip,jp)),shape=(numNodes,numVerts))
Dx = coo_matrix((dx_data,(ipd,jpd)),shape=(numVerts,numNodes))
Dy = coo_matrix((dy_data,(ipd,jpd)),shape=(numVerts,numNodes))

K  = K.tolil() # K11 = K22
M  = M.tolil() # M11 = M22
#Cx = Cx.tolil()
#Cy = Cy.tolil()
Gx = Gx.tolil()
Gy = Gy.tolil()
Dx = Dx.tolil()
Dy = Dy.tolil()

end = time.time()
assembly_t = str(round(end-start,3))
# -----------------------------------------------------------------------------
