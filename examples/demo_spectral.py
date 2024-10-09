import fastfem.elements.spectral_element as SE
import numpy as np
import argparse

def run_demo(f_generator,elem_order:int, nelem_x:int, nelem_y:int,
             max_iters:int = 50):
    """Runs a spectral element method to solve the poisson equation
    on Ω=(0,1)² with homogeneous Dirichlet (essential) boundary conditions.
    Given the basis V generated by the spectral elements, we find u ∈ V so that
    { A(u,v) + f(v) = 0 ∀ v ∈ V } where A(u,v) is the bilinear form
    { ∫ ∇u • ∇v dx^2 }.
    
    f_generator is called for each element as
        f_generator(element_object, points,testfunc) -> float
        - element_object (SpectralElement2D): instance of the element used
        - points (ndarray): (num_nodes,num_nodes,2) array of positions
        - testfunc (ndarray): (num_nodes,num_nodes) array representing the test
            function in the local basis
    f(v) is taken to be the sum of f_generator return values. It is assumed,
    from linearity, that f_generator() returns 0 if all(testfunc == 0), and
    is therefore not called.
    
    The linear equation is solved using an iterative method.

    Args:
        f_generator (function): the generator function for the functional f.
        elem_order (int): the degree of the polynomial used in the element
        nelem_x (int): how many elements along the x-axis
        nelem_y (int): how many elements along the y-axis
        max_iters (int, optional): The maximum number of iterations for the
            solver. Defaults to 5.
    """
    elem = SE.SpectralElement2D(elem_order)
    
    #each point in each element is assigned a global index (assembly)
    #that matches neighboring elements across boundaries
    
    global_indices,global_dim = build_assembly(elem_order,nelem_x,nelem_y)
    cell_width = 1/nelem_x
    cell_height = 1/nelem_y
    coords = np.empty((nelem_x,nelem_y,elem_order+1,elem_order+1,2))
    
    #holder for basis function
    testfield = np.empty((elem_order+1,elem_order+1))
    
    #array of f(v) on global basis with throw-away value for boundaries
    F = np.zeros(global_dim+1)
    
    for i in range(nelem_x):
        for j in range(nelem_y):
            coords[i,j,:,:,0] = cell_width *(i+(elem.knots[:,np.newaxis]+1)/2)
            coords[i,j,:,:,1] = cell_height*(j+(elem.knots[np.newaxis,:]+1)/2)
            for k in range(elem_order+1):
                for l in range(elem_order+1):
                    gid = global_indices[i,j,k,l]
                    if gid >= 0:
                        testfield[...] = 0
                        testfield[k,l] = 1
                        F[gid] += f_generator(elem,coords[i,j],testfield)
    
    #u in global basis with additional throw-away value for boundaries
    U = np.zeros(global_dim+1)
    
    U_TRUE = np.zeros(global_dim+1)
    for i in range(nelem_x):
        for j in range(nelem_y):
            U_TRUE[global_indices[i,j]] = u_true(coords[i,j,:,:,0],coords[i,j,:,:,1])
    U_TRUE[-1] = 0
    U[:] = U_TRUE
    
    A = np.zeros((global_dim+1,global_dim+1))
    #we are trying to solve AU = F, where A is a sum of sparse blocks
    #use Jacobi
    for iter in range(max_iters):
        #b - (A-diag(A)) @ U  ~~  diag(A) @ U; we divide after by DIAG
        STEP = -F
        DIAG = np.zeros(global_dim+1)
        
        for i in range(nelem_x):
            for j in range(nelem_y):
                STEP[global_indices[i,j]] -= \
                    elem.basis_stiffness_matrix_times_field(coords[i,j],
                        U[global_indices[i,j]]
                    )
                DIAG[global_indices[i,j]] += \
                    elem.basis_stiffness_matrix_diagonal(coords[i,j])
        
        STEP = STEP[:-1]/DIAG[:-1]
        #under-relaxation, take substep:
        U[:-1] += 0.67*STEP
        print(iter,np.linalg.norm(U-U_TRUE,ord=np.inf),np.linalg.norm(STEP,ord=np.inf))
        if np.linalg.norm(STEP,ord=np.inf) < 1e-9:
            break
    u = np.empty((nelem_x,nelem_y,elem_order+1,elem_order+1))
    for i in range(nelem_x):
        for j in range(nelem_y):
            u[i,j,:,:] = U[global_indices[i,j]]
    return coords,u
    
                
                

def build_assembly(elem_order,nelem_x,nelem_y):
    global_indices = np.zeros((nelem_x,nelem_y,elem_order+1,elem_order+1),
                              dtype=int)
    global_size = 0
    for i in range(nelem_x):
        for j in range(nelem_y):
            #every element has the interior
            global_indices[i,j,1:-1,1:-1] = global_size \
                + np.arange((elem_order-1)**2)\
                    .reshape((elem_order-1,elem_order-1))
            global_size += (elem_order-1)**2
            
            # -x
            if i == 0:
                #exclude from global basis
                global_indices[i,j,0,:] = -1
            else:
                #continuous with left elem
                global_indices[i,j,0,:] = global_indices[i-1,j,-1,:]
            
            # +x
            if i == nelem_x - 1:
                #exclude from global basis
                global_indices[i,j,-1,:] = -1
            else:
                #new indices
                global_indices[i,j,-1,1:-1] = global_size + np.arange(
                    elem_order-1)
                global_size += elem_order-1
                if j < nelem_y - 1:
                    global_indices[i,j,-1,-1] = global_size
                    global_size += 1
                    
            # -y
            if j == 0:
                #exclude from global basis
                global_indices[i,j,:,0] = -1
            else:
                #continuous with bottom elem
                global_indices[i,j,:,0] = global_indices[i,j-1,:,-1]
            
            # +y
            if j == nelem_y - 1:
                #exclude from global basis
                global_indices[i,j,:,-1] = -1
            else:
                #new indices
                global_indices[i,j,1:-1,-1] = global_size + np.arange(
                    elem_order-1)
                global_size += elem_order-1
    return global_indices,global_size

def build_argparse():
    parser = argparse.ArgumentParser(
        prog="demo_spectral",
        description="""
Example code for the spectral elements to solve the Poisson Equation.
This code uses a custom mesh assembly, separate from the package.
                    """,
        epilog="""
The equation
        { Δu = f on Ω=(0,1)²   and   u = 0 on ∂Ω }
is solved using the analytic solution [TODO write demo true sol].
        """)
    parser.add_argument("-o","--order",help="""
Element order (degree of the polynomial to be used). The number of
nodes per axis
    """,action="store",default=5,type=int,metavar="ELEM_ORDER")
    parser.add_argument("-x","--nelemx","--nx",help="""
Number of elements along the x-axis
    """,action="store",default=10,type=int,metavar="NELEM_X")
    parser.add_argument("-y","--nelemy","--ny",help="""
Number of elements along the y-axis
    """,action="store",default=10,type=int,metavar="NELEM_Y")
    parser.add_argument("-p","--plot",help="""
Set to plot using matplotlib.
    """,action="store_true")
    return parser

if __name__ == "__main__":
    args = build_argparse().parse_args()
    print(f"Demo-ing spectral elements: order {args.order} on a "+
          f"{args.nelemx}x{args.nelemy} grid")
    u_true = lambda x,y: np.sin(np.pi * x) * y * (1-y)
    def f_generator(elem,points,testfunc):
        u_lapl = lambda x,y: -np.pi**2*np.sin(np.pi * x)*y*(1-y) + np.sin(np.pi * x) * (1 - 2*y)
        # # use midpoint rule at high resolution
        # N_ = 20
        # h = 1/N_
        # MIDPOINTS = h*(np.arange(N_)+0.5)
        # J = np.linalg.det(elem.def_grad(points,MIDPOINTS[:,np.newaxis],MIDPOINTS))
        
        # return np.sum(u_lapl(MIDPOINTS[:,np.newaxis],MIDPOINTS)
        #               * elem.interp_field(testfunc,MIDPOINTS[:,np.newaxis],MIDPOINTS) * J) * (h**2)
        
        #just use the spectral quadrature
        return np.einsum("ij->",elem.basis_mass_matrix(points)*testfunc*u_lapl(points[:,:,0],points[:,:,1]))
    coords, u = run_demo(f_generator,args.order,args.nelemx,args.nelemy,max_iters=500)
    u_comp = u_true(coords[...,0],coords[...,1])
    print("infinity norm error",np.max(np.abs(u - u_comp)))
    import matplotlib.pyplot as plt
    plt.figure(0)
    plt.scatter(coords[...,0],coords[...,1],10,u-u_comp)
    plt.show()