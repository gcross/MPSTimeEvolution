#@+leo-ver=4-thin
#@+node:gmc.20080805172037.20:@thin time_evolution_transverse_ising.py
#@<< Import needed modules >>
#@+node:gmc.20080805172037.48:<< Import needed modules >>
from numpy import *
from scipy.linalg import *
#@-node:gmc.20080805172037.48:<< Import needed modules >>
#@nl

#@+others
#@+node:gmc.20080805172037.73:Exceptions
#@+others
#@+node:gmc.20080805172037.74:AllVectorsVanished
class AllVectorsVanished(Exception):
    """Raised when all vectors vanish in the SVD performed inside merge_and_split."""
    pass
#@-node:gmc.20080805172037.74:AllVectorsVanished
#@-others
#@-node:gmc.20080805172037.73:Exceptions
#@+node:gmc.20080805172037.49:Functions
#@+others
#@+node:gmc.20080805172037.50:def build_ZZ_MPO
def build_ZZ_MPO(dt):

#@-node:gmc.20080805172037.50:def build_ZZ_MPO
#@+node:gmc.20080805172037.72:merge_and_split
def merge_and_split(A,A_index,B,B_index,compressed_dimension):
    """Merge and split performs bond reduction between two tensors by merging them
together and then splitting them apart using SVD.  You need to specify the two
tensors, the indices along which they are joined, and the final (maximum) size of
the bond.  (The routine may return tensors with a smaller bond, if some of the
singular values vanish.)

Usage:

    A, lambda_, B = merge_and_split(A,A_index,B,B_index,compressed_dimension)
"""

    # Merge the two tensors together
    AB = tensordot(A,B,(A_index,B_index))

    # Reshape the merged tensor into a matrix
    M = AB.reshape(prod(A.shape)/A.shape[A_index],prod(B.shape)/B.shape[B_index])

    # Split them apart using SVD
    u, s, v = svd(M,full_matrices=0,overwrite_a=1)

    # Perform initial truncation
    s = s[:compressed_dimension]

    # Normalization
    s /= s[0]

    # Additionally, truncate all zero singular values
    cutoff = 0
    while cutoff < len(s) and abs(s[cutoff]) > 1e-14:
        cutoff += 1

    # If *all* the singular values were zero, then something is wrong.
    if cutoff == 0:
        raise AllVectorsVanished

    # Truncate unwanted vectors/singular values
    u = u[:,:cutoff]
    s = s[:cutoff]
    v = v[:cutoff,:].transpose()

    def final_processing(m,M,M_index):
        """Reshapes and transposes the tensor to return it to its original form."""
        new_shape = list(M.shape)
        del new_shape[M_index]
        new_shape.append(cutoff)
        new_indices = range(len(new_shape)-1)
        new_indices.insert(M_index,len(new_shape)-1)

        return m.reshape(new_shape).transpose(new_indices)

    # Reshape and transpose tensors to return to their original form
    new_u = final_processing(u,A,A_index)
    new_v = final_processing(v,B,B_index)

    # We're done!  Return the result.
    return new_u, s, new_v


#@-node:gmc.20080805172037.72:merge_and_split
#@-others
#@-node:gmc.20080805172037.49:Functions
#@-others

#@<< Initialization >>
#@+node:gmc.20080805172037.77:<< Initialization >>
#@<< Set parameters >>
#@+node:gmc.20080805172037.78:<< Set parameters >>
#@+at
# Length of time step
#@-at
#@@c
dt = 0.01

#@+at
# Number of time steps to take before increasing J to its next value.
#@-at
#@@c
number_of_time_steps_per_J = 10

#@+at
# Parameters describing how you want J to evolve.
#@-at
#@@c
initial_J_value = 0
final_J_value = 1
J_step = 0.001

#@+at
# Number of sites in the system.
#@-at
#@@c
number_of_sites = 10

#@+at
# The size to which bonds should be truncated.
#@-at
#@@c
compressed_dimension = 2
#@-node:gmc.20080805172037.78:<< Set parameters >>
#@nl

#@<< Build data structure for MPS >>
#@+node:gmc.20080805172037.79:<< Build data structure for MPS >>
#@+at
# 
#@-at
#@@c
site_tensors = [ones(2,1)] + [ones(2,1,1) for dummy in xrange(number_of_sites-2)] + [ones(2,1)]
lambdas = [ones(1,1)] + [ones(1,1,1) for dummy in xrange(number_of_sites-2)]
#@-node:gmc.20080805172037.79:<< Build data structure for MPS >>
#@nl
#@-node:gmc.20080805172037.77:<< Initialization >>
#@nl

#@<< Main loop >>
#@+node:gmc.20080805172037.76:<< Main loop >>
#@+at
# Note:  arange(X,Y) returns values including X but excluding Y;  this
#        why I add 1e-10 to final_J_value -- to make sure that the right
#        endpoint is included.
#@-at
#@@c
for J in arange(initial_J_value,final_J_value+1e-10,J_step):
    #@    << Build MPO >>
    #@+node:gmc.20080805172037.80:<< Build MPO >>
    #@-node:gmc.20080805172037.80:<< Build MPO >>
    #@nl

    #@    << Apply unitary >>
    #@+node:gmc.20080805172037.75:<< Apply unitary >>
    #@+at
    # The tensors at the boundaries are a special case, so we handle them 
    # separately.
    # 
    # First, absorb the lambdas into the site tensors.  Note that the lambdas 
    # are
    # assumed to be shaped so that they get multiplied into the correct index.
    #@-at
    #@@c
    site_tensors[0] *= lambdas[0]
    site_tensors[1] *= lambdas[1]

    #@+at
    # Apply the MPO to these site tensors.
    #@-at
    #@@c
    site_tensors[0] = multiply_left_site_tensor_by_operator_tensor(site_tensor[0],left_operator_tensor)
    site_tensors[1] = multiply_interior_site_tensor_by_operator_tensor(site_tensor[1],interior_operator_tensor)

    #@+at
    # Reduce the size of the bond between the two tensors.
    #@-at
    #@@c
    site_tensors[0], lambda_, site_tensors[1] = merge_and_split(site_tensor[0],1,site_tensor[1],1,compressed_dimension)

    #@+at
    # Reshape the lambda so that it lines up with the correct index in
    # site_tensor[0].
    #@-at
    #@@c
    lambdas[0] = lambda_.reshape(1,lambda_.shape[0])

    #@+at
    # Now we handle the interior tensors.  This is essentially the same idea 
    # as before, so
    # step-by-step commentary will not be repeated.
    #@-at
    #@@c
    for i in xrange(1,number_of_sites-1):
        site_tensors[i+1] *= lambdas[i+1]
        site_tensors[i+1] = multiply_interior_site_tensor_by_operator_tensor(site_tensor[i+1],interior_operator_tensor)
        site_tensors[i], lambda_, site_tensor[i+1] = merge_and_split(site_tensor[i],2,site_tensor[i+1],1,compressed_dimension)
        lambdas[i] = lambda_.reshape(1,lambda_.shape[0],1)

    #@+at
    # Finally, we handle the right boundary.  Note that there is no lambda to 
    # the right
    # of the right boundary, and hence we skip the lambda absorption step.
    #@-at
    #@@c
    site_tensors[-1] = multiply_right_site_tensor_by_operator_tensor(site_tensor[i+1],right_operator_tensor)
    site_tensors[-2], lambda_, site_tensors[-1] = merge_and_split(site_tensor[-2],2,site_tensor[-1],1,compressed_dimension)
    lambdas[i] = lambda_.reshape(1,lambda_.shape[0],1)
    #@nonl
    #@-node:gmc.20080805172037.75:<< Apply unitary >>
    #@nl
#@-node:gmc.20080805172037.76:<< Main loop >>
#@nl
#@nonl
#@-node:gmc.20080805172037.20:@thin time_evolution_transverse_ising.py
#@-leo
