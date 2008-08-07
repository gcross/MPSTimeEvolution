#@+leo-ver=4-thin
#@+node:gmc.20080805172037.20:@thin time_evolution_transverse_ising.py
#@<< Import needed modules >>
#@+node:gmc.20080805172037.48:<< Import needed modules >>
from __future__ import division  # have division return a float by default, even between integers
import __builtin__ # needed so we can invoke builting Python functions shadowed by NumPy functions with the same name

from numpy import *
from scipy.linalg import *
#@-node:gmc.20080805172037.48:<< Import needed modules >>
#@nl

#@<< Define Pauli operators >>
#@+node:gmc.20080806124136.2:<< Define Pauli operators >>
#@+at
# Construct matrices for the (unnormalized) Pauli operators.
#@-at
#@@c
I = array([[1,0],[0,1]],complex128)
X = array([[0,1],[1,0]],complex128)
Y = array([[0,1j],[-1j,0]],complex128)
Z = array([[1,0],[0,-1]],complex128)

#@+at
# For convenience and performance, we construct copies of the Pauli
# operators that have been premultiplied by -i:
#@-at
#@@c
miI = -1j*I
miX = -1j*X
miY = -1j*Y
miZ = -1j*Z
#@-node:gmc.20080806124136.2:<< Define Pauli operators >>
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
#@+node:gmc.20080806124136.12:Contractors
#@+at
# This section of the code creates macros which performs various tensor
# network contractions needed to implement the algorithm.  This is done by
# using a nested list to describe the structure of the network, and then
# specifying the "output" indices, tuple-ing indices which should be grouped
# together into a single index.  This specification is passed to the routine
# make_contractor_from_implicit_joins, which builds a function that contracts
# the network.
#@-at
#@@c

#@<< Macro building functions >>
#@+node:gmc.20080806124136.13:<< Macro building functions >>
#@+at
# These routines build macros to perform tensor contractions.  They do this
# by construction a string of Python code which performs the contraction,
# and then compiling the code into a function.
#@-at
#@@c

#@+others
#@+node:gmc.20080806124136.24:n2l
#@+at
# Utility function converting numbers to letters.
#@-at
#@@c
n2l = map(chr,range(ord('A'),ord('Z')+1))
#@-node:gmc.20080806124136.24:n2l
#@+node:gmc.20080806124136.6:make_contractor
def make_contractor(tensor_index_labels,index_join_pairs,result_index_labels,name="f"):    # pre-process parameters
    tensor_index_labels = list(map(list,tensor_index_labels))
    index_join_pairs = list(index_join_pairs)
    result_index_labels = list([list(index_group) if hasattr(index_group,"__getitem__") else [index_group] for index_group in result_index_labels])

    assert sum(len(index_group) for index_group in tensor_index_labels) == (sum(len(index_group) for index_group in result_index_labels)+2*len(index_join_pairs))

    function_definition_statements = ["def %s(%s):" % (name,",".join(n2l[:len(tensor_index_labels)]))]

    #@    << def build_statements >>
    #@+node:gmc.20080806124136.7:<< def build_statements >>
    def build_statements(tensor_index_labels,index_join_pairs,result_index_labels):
    #@+at
    # This routine recursively builds a list of statements which performs the 
    # full tensor contraction.
    # 
    # First, if there is only one tensor left, then transpose and reshape it 
    # to match the result_index_labels.
    #@-at
    #@@c
        if len(tensor_index_labels) == 1:
            if len(result_index_labels) == 0:
                return ["return A"]
            else:
                final_index_labels = tensor_index_labels[0]
                result_indices = [[final_index_labels.index(index) for index in index_group] for index_group in result_index_labels]
                transposed_indices = __builtin__.sum(result_indices,[])
                assert type(transposed_indices) == list
                assert len(final_index_labels) == len(transposed_indices)
                new_shape = ",".join(["(%s)" % "*".join(["shape[%i]"%index for index in index_group]) for index_group in result_indices])     
                return ["shape=A.shape","return A.transpose(%s).reshape(%s)" % (transposed_indices,new_shape)]
    #@+at
    # Second, if all joins have finished, then take outer products to combine 
    # all remaining tensors into one.
    #@-at
    #@@c
        elif len(index_join_pairs) == 0:
            if tensor_index_labels[-1] is None:
                return build_statements(tensor_index_labels[:-1],index_join_pairs,result_index_labels)
            elif len(tensor_index_labels[-1]) == 0:
                v = n2l[len(tensor_index_labels)-1]
                return ["A*=%s" % v, "del %s" % v] + build_statements(tensor_index_labels[:-1],index_join_pairs,result_index_labels)
            else:
                v = n2l[len(tensor_index_labels)-1]
                tensor_index_labels[0] += tensor_index_labels[-1]
                return ["A = multiply.outer(A,%s)" % v, "del %s" % v] + build_statements(tensor_index_labels[:-1],index_join_pairs,result_index_labels)
    #@+at
    # Otherwise, do the first join, walking through index_join_pairs to find 
    # any other pairs which connect the same two tensors.
    #@-at
    #@@c
        else:
            #@        << Search for all joins between these tensors >>
            #@+node:gmc.20080806124136.8:<< Search for all joins between these tensors >>
            #@+at
            # This function searches for the tensors which are joined, and 
            # reorders the indices in the join so that the index corresponding 
            # to the tensor appearing first in the list of tensors appears 
            # first in the join.
            #@-at
            #@@c
            def find_tensor_ids(join):
                reordered_join = [None,None]
                tensor_ids = [0,0]
                join = list(join)
                while tensor_ids[0] < len(tensor_index_labels):
                    index_labels = tensor_index_labels[tensor_ids[0]]
                    if index_labels is None:
                        tensor_ids[0] += 1
                    elif join[0] in index_labels:
                        reordered_join[0] = index_labels.index(join[0])
                        del join[0]
                        break
                    elif join[1] in index_labels:
                        reordered_join[0] = index_labels.index(join[1])
                        del join[1]
                        break
                    else:
                        tensor_ids[0] += 1
                assert len(join) == 1 # otherwise index was not found in any tensor
                tensor_ids[1] = tensor_ids[0] + 1
                while tensor_ids[1] < len(tensor_index_labels):
                    index_labels = tensor_index_labels[tensor_ids[1]]
                    if index_labels is None:
                        tensor_ids[1] += 1
                    elif join[0] in index_labels:
                        reordered_join[reordered_join.index(None)] = index_labels.index(join[0])
                        del join[0]
                        break
                    else:
                        tensor_ids[1] += 1
                assert len(join) == 0 # otherwise index was not found in any tensor
                return tensor_ids, reordered_join

            join_indices = [0]
            tensor_ids,reordered_join = find_tensor_ids(index_join_pairs[0])

            indices = [[],[]]

            for j in xrange(2):
                indices[j].append(reordered_join[j])

            # Search for other joins between these tensors
            for i in xrange(1,len(index_join_pairs)):
                tensor_ids_,reordered_join = find_tensor_ids(index_join_pairs[i])
                if tensor_ids == tensor_ids_:
                    join_indices.append(i)
                    for j in xrange(2):
                        indices[j].append(reordered_join[j])

            #@-node:gmc.20080806124136.8:<< Search for all joins between these tensors >>
            #@nl

            #@        << Build tensor contraction statements >>
            #@+node:gmc.20080806124136.9:<< Build tensor contraction statements >>
            tensor_vars = [n2l[id] for id in tensor_ids]

            statements = [
                "try:",
                "   %s = tensordot(%s,%s,%s)" % (tensor_vars[0],tensor_vars[0],tensor_vars[1],indices),
                "   del %s" % tensor_vars[1],
                "except ValueError:",
                "   raise ValueError('indices %%s do not match for tensor %%i, shape %%s, and tensor %%i, shape %%s.' %% (%s,%i,%s.shape,%i,%s.shape))" % (indices,tensor_ids[0],tensor_vars[0],tensor_ids[1],tensor_vars[1])
            ]
            #@-node:gmc.20080806124136.9:<< Build tensor contraction statements >>
            #@nl

            #@        << Delete joins from list and update tensor specifications >>
            #@+node:gmc.20080806124136.10:<< Delete joins from list and update tensor specifications >>
            join_indices.reverse()
            for join_index in join_indices:
                del index_join_pairs[join_index]

            new_tensor_index_labels_0 = list(tensor_index_labels[tensor_ids[0]])
            indices[0].sort(reverse=True)
            for index in indices[0]:
                del new_tensor_index_labels_0[index]

            new_tensor_index_labels_1 = list(tensor_index_labels[tensor_ids[1]])
            indices[1].sort(reverse=True)
            for index in indices[1]:
                del new_tensor_index_labels_1[index]

            tensor_index_labels[tensor_ids[0]] = new_tensor_index_labels_0+new_tensor_index_labels_1
            tensor_index_labels[tensor_ids[1]] = None
            #@-node:gmc.20080806124136.10:<< Delete joins from list and update tensor specifications >>
            #@nl

            return statements + build_statements(tensor_index_labels,index_join_pairs,result_index_labels)
    #@-node:gmc.20080806124136.7:<< def build_statements >>
    #@nl

    function_definition_statements += ["\t" + statement for statement in build_statements(tensor_index_labels,index_join_pairs,result_index_labels)]

    function_definition = "\n".join(function_definition_statements)+"\n"

    f_globals = {"tensordot":tensordot,"multiply":multiply}
    f_locals = {}

    exec function_definition in f_globals, f_locals

    f = f_locals[name]
    f.source = function_definition
    return f
#@nonl
#@-node:gmc.20080806124136.6:make_contractor
#@+node:gmc.20080806124136.11:make_contractor_from_implicit_joins
def make_contractor_from_implicit_joins(tensor_index_labels,result_index_labels,name="f"):
    tensor_index_labels = list(map(list,tensor_index_labels))
    found_indices = {}
    index_join_pairs = []
    for i in xrange(len(tensor_index_labels)):
        for index_position, index in enumerate(tensor_index_labels[i]):
            if index in found_indices:
                other_tensor = found_indices[index]
                if other_tensor is None:
                    raise ValueError("index label %s found in more than two tensors" % index)
                else:
                    # rename this instance of the index and add to the list of join pairs
                    tensor_index_labels[i][index_position] = (i,index)
                    index_join_pairs.append((index,(i,index)))
                    # mark that we have found two instances of this index for
                    # error-checking purposes
                    found_indices[index] = None
            else:
                found_indices[index] = i
    return make_contractor(tensor_index_labels,index_join_pairs,result_index_labels,name)
#@nonl
#@-node:gmc.20080806124136.11:make_contractor_from_implicit_joins
#@-others
#@-node:gmc.20080806124136.13:<< Macro building functions >>
#@nl

#@+others
#@+node:gmc.20080806124136.16:Contractors for applying operators to sites
#@+others
#@+node:gmc.20080806124136.14:multiply_left/right_site_tensor_by_operator_tensor
#@+at
# Constracts S and O together to form the new tensor N, grouping together the
# right indices 2 and 12.
# 
# S -2-
# |
# +1
# |
# O -12-
# |
# -1
# |
# 
# =
# 
# N -- 2-
# | \-12-
# |
# -1
# |
# 
#@-at
#@@c

multiply_left_site_tensor_by_operator_tensor = make_contractor_from_implicit_joins([
    [1,2],     # indices of the site tensor
    [12,-1,1], # indices of the operator tensor
],[
    -1,
    (2,12) # group together auxiliary indices
])

#@+at
# If you analyize the contraction for the right tensor,
# 
#  - 2-S
#      |
#     +1
#      |
#  -12-O
#      |
#     -1
#      |
# 
# =
# 
#  - 2-- N
#  -12-/ |
#        |
#       -1
#        |
# 
# you see that it ends up being exactly the same as the
# contraction for the left tensor, since the indices are
# all in the same places on the two tensors.  Thus, we
# duplication the left contraction function to form the
# right contraction function.
# 
#@-at
#@@c

multiply_right_site_tensor_by_operator_tensor = multiply_left_site_tensor_by_operator_tensor
#@-node:gmc.20080806124136.14:multiply_left/right_site_tensor_by_operator_tensor
#@+node:gmc.20080806124136.15:multiply_interior_site_tensor_by_operator_tensor
#@+at
# Constracts S and O together to form the new tensor N, grouping together the
# left indices 2 and 12 and right indices 3 and 13.
# 
#  - 2-S- 3-
#      |
#     +1
#      |
#  -12-O-13-
#      |
#     -1
#      |
# 
# =
# 
#  - 2-- N -- 3-
#  -12-/ | \-13-
#        |
#       -1
#        |
# 
# 
#@-at
#@@c

multiply_interior_site_tensor_by_operator_tensor = make_contractor_from_implicit_joins([
    [1,2,3],      # indices of the site tensor
    [12,13,-1,1], # indices of the operator tensor
],[
    -1,
    (2,12), # group together left auxiliary indices
    (3,13)  # group together right auxiliary indices
])
#@-node:gmc.20080806124136.15:multiply_interior_site_tensor_by_operator_tensor
#@-others
#@-node:gmc.20080806124136.16:Contractors for applying operators to sites
#@+node:gmc.20080806124136.17:Contractors used in computing expected values
#@+others
#@+node:gmc.20080806124136.18:form_left_boundary
#@+at
# Forms the left boundary tensor used in computing expected values of 
# operators.
# 
# S--2
# |
# 1
# |
# O--12
# |
# -1
# |
# S*-22
# 
# =
#    2
#   /
#  /
# N--12
#  \
#   \
#    22
# 
#@-at
#@@c

_form_left_boundary = make_contractor_from_implicit_joins([
    [1,2],     # indices of the site tensor
    [12,-1,1], # indices of the operator tensor
    [-1,22],   # indices of the conjugated tensor
],[
    2,
    12,
    22
])

def form_left_boundary(S,lambda_,O):
    S = S*lambda_
    return _form_left_boundary(S,O,S.conj())
#@-node:gmc.20080806124136.18:form_left_boundary
#@+node:gmc.20080806124136.19:absorb_interior_site_into_left_boundary
#@+at
# Absorbs the site tensor and operator at a site into the left boundary;
# used in compute expected values of operators.
# 
# 
#     -S--3
#    / |
#   2  1
#  /   |
# L-12-O--13
#  \   |
#   32 -1
#    \ |
#     -S*-23
# 
# =
#    3
#   /
#  /
# N--13
#  \
#   \
#    23
# 
#@-at
#@@c

_absorb_interior_site_into_left_boundary = make_contractor_from_implicit_joins([
    [2,12,22],    # indices of the left boundary tensor
    [1,2,3],      # indices of the site tensor
    [12,13,-1,1], # indices of the operator tensor
    [-1,22,23],   # indices of the conjugated tensor
],[
    3,
    13,
    23
])

def absorb_interior_site_into_left_boundary(L,S,lambda_,O):
    S = S*lambda_
    return _absorb_interior_site_into_left_boundary(L,S,O,S.conj())
#@-node:gmc.20080806124136.19:absorb_interior_site_into_left_boundary
#@+node:gmc.20080806124136.20:merge_left_boundary_with_right_boundary
#@+at
# Finishes the expected value contraction merging the right boundary site
# and operator tensor with the left boundary.
# 
# 
#     -S
#    / |
#   2  1
#  /   |
# L-12-O
#  \   |
#   32 -1
#    \ |
#     -S*
# 
# =
# 
# N (scalar)
# 
#@-at
#@@c

_merge_left_boundary_with_right_boundary = make_contractor_from_implicit_joins([
    [2,12,22],# indices of the left boundary tensor
    [1,2],     # indices of the site tensor
    [12,-1,1], # indices of the operator tensor
    [-1,22],   # indices of the conjugated tensor
],[()] # indicates that a scalar should be returned
)

def merge_left_boundary_with_right_boundary(L,S,O):
    return _merge_left_boundary_with_right_boundary(L,S,O,S.conj())
#@-node:gmc.20080806124136.20:merge_left_boundary_with_right_boundary
#@-others
#@-node:gmc.20080806124136.17:Contractors used in computing expected values
#@-others
#@-node:gmc.20080806124136.12:Contractors
#@+node:gmc.20080805172037.49:Functions
#@+others
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

    # Additionally, truncate all singular values close to zero
    cutoff = 0
    while cutoff < len(s) and abs(s[cutoff]) > 1e-7:
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
#@+node:gmc.20080806124136.5:multiply_tensor_by_matrix_at_index
def multiply_tensor_by_matrix_at_index(tensor,matrix,index):
    """This function dots the given matrix into the tensor at the given index,
automatically taking care of rearranging the indices so that they end up the
same order as when they started."""
    tensor_new_indices = range(tensor.ndim-1)
    tensor_new_indices.insert(index,tensor.ndim-1)
    return tensordot(tensor,matrix,(index,0)).transpose(tensor_new_indices)
#@-node:gmc.20080806124136.5:multiply_tensor_by_matrix_at_index
#@+node:gmc.20080806124136.21:compute_expected_value
def compute_expected_value(left_operator_tensor,interior_operator_tensor,right_operator_tensor):
    left_boundary = form_left_boundary(site_tensors[0],lambdas[0],left_operator_tensor)
    for i in xrange(1,number_of_sites-1):
        left_boundary = absorb_interior_site_into_left_boundary(left_boundary,site_tensors[i],lambdas[i],interior_operator_tensor)
    return merge_left_boundary_with_right_boundary(left_boundary,site_tensors[-1],right_operator_tensor)
#@-node:gmc.20080806124136.21:compute_expected_value
#@+node:gmc.20080806124136.22:compute_energy
def compute_energy(J):

    #@    << Build matrix product operator representation of the Hamiltonian >>
    #@+node:gmc.20080806124136.23:<< Build matrix product operator representation of the Hamiltonian >>
    J = final_J_value

    #@+at
    # The following is the matrix product operator representation of the 
    # transverse Ising Hamiltonian.
    # For a detailed discussion of how to do this for operators in general, 
    # see arXiv:0708.1221,
    # "Finite automata for caching in matrix product algorithms", Crosswhite & 
    # Bacon.
    # 
    # At the left boundary, the finite state automaton can choose to output 
    # either a I, an Z, or a -X.
    # It does this, and sends a signal about its choice (respectively: 0, 1, 
    # or 2) to the right.
    #@-at
    #@@c

    left_hamiltonian_tensor = array([I,Z,-X])

    #@+at
    # At the right boundary, there are three posibilities, corresponding to 
    # the following signals:
    # 
    #     0) no ZZ or X has been placed on my left, so place a -X here
    #     1) a Z has been placed directly to my left, so put a -J*Z here
    #     2) a ZZ or an X has been placed somewhere to the left, so put an I 
    # here
    #@-at
    #@@c

    right_hamiltonian_tensor = array([-X,-J*Z,I])

    #@+at
    # In the interior, the following input->output patterns are possible
    # 
    #     Input 0:  nothing but I's to the left;  make a choice
    #     0->0 I
    #     0->1 Z
    #     0->2 -X
    # 
    #     Input 1:  a Z directly on my left;  no choice -- put a -J*Z here
    #     1->2 -J*Z
    # 
    #     Input 2:  a ZZ or X somewhere on the left;  no choice -- put an I 
    # here
    #     2->2 I
    # 
    #@-at
    #@@c

    interior_hamiltonian_tensor = zeros((3,3,2,2),complex128)

    interior_hamiltonian_tensor[0,0] = I
    interior_hamiltonian_tensor[0,1] = Z
    interior_hamiltonian_tensor[0,2] = -X

    interior_hamiltonian_tensor[1,2] = -J*Z

    interior_hamiltonian_tensor[2,2] = I
    #@-node:gmc.20080806124136.23:<< Build matrix product operator representation of the Hamiltonian >>
    #@nl

    #@    << Built matrix product operator representation of the Identity >>
    #@+node:gmc.20080806124136.25:<< Built matrix product operator representation of the Identity >>
    left_identity_tensor = identity(2).reshape(1,2,2)
    interior_identity_tensor = identity(2).reshape(1,1,2,2)
    right_identity_tensor = identity(2).reshape(1,2,2)
    #@-node:gmc.20080806124136.25:<< Built matrix product operator representation of the Identity >>
    #@nl

    energy = compute_expected_value(left_hamiltonian_tensor,interior_hamiltonian_tensor,right_hamiltonian_tensor)
    normalizer = compute_expected_value(left_identity_tensor,interior_identity_tensor,right_identity_tensor)

    return energy/normalizer
#@-node:gmc.20080806124136.22:compute_energy
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
dt = 0.001

#@+at
# Number of time steps to take before increasing J to its next value.
#@-at
#@@c
number_of_time_steps_per_J = 1

#@+at
# Parameters describing how you want J to evolve.
#@-at
#@@c
initial_J_value = 0
final_J_value = 0.5
J_step = 0.0001

#@+at
# Number of sites in the system.
#@-at
#@@c
number_of_sites = 12

#@+at
# The size to which bonds should be truncated.
#@-at
#@@c
compressed_dimension = 16
#@-node:gmc.20080805172037.78:<< Set parameters >>
#@nl

#@<< Build data structure for MPS >>
#@+node:gmc.20080805172037.79:<< Build data structure for MPS >>
#@+at
# Initialize the matrix product state to the (unnormalized) + state -- i.e., 
# the
# outer product of the array [1,1] at all sites.  The inner bond dimensions 
# are
# set to size 1.  They will grow automatically as we time evolve the system.
#@-at
#@@c

initial_state = array([1,1],complex128)

site_tensors = [initial_state.copy().reshape(2,1)] \
             + [initial_state.copy().reshape(2,1,1) for dummy in xrange(number_of_sites-2)] \
             + [initial_state.copy().reshape(2,1)]

lambdas = [ones((1,1),complex128)] + [ones((1,1,1),complex128) for dummy in xrange(number_of_sites-2)]
#@-node:gmc.20080805172037.79:<< Build data structure for MPS >>
#@nl
#@-node:gmc.20080805172037.77:<< Initialization >>
#@nl

#@<< Main loop >>
#@+node:gmc.20080805172037.76:<< Main loop >>
total_number_of_J_steps = int((final_J_value-initial_J_value)/J_step)
counter = 0

J = initial_J_value
print "Start:  %f:%f" % (J,compute_energy(J))

print "Performing adiabatic evolution..."

#@+at
# Note:  arange(X,Y) returns values including X but excluding Y;  this
#        why I add 1e-10 to final_J_value -- to make sure that the right
#        endpoint is included.
#@-at
#@@c
for J in arange(initial_J_value,final_J_value+1e-10,J_step):
    #@    << Build MPO >>
    #@+node:gmc.20080805172037.80:<< Build MPO >>
    #@+at
    # We want to apply the unitary
    # 
    #     exp(sum_k (-itX_k) + sum_k (-itZ_kZ_k))
    # 
    # However, to compute and apply this exactly would be expensive, so 
    # instead
    # we construct a Trotter approximation of the unitary,
    # 
    #     exp(sum_k (-itX_k)/2) exp(sum_k (-itZ_kZ_k)) exp(sum_k (-itX_k)/2)
    # 
    # which is accurate to O(k).
    # 
    # To do this, we first construct a matrix product operator for
    # exp(sum_k (-itZ_kZ_k)), and then multiply this operator on both sides
    # by exp(sum_k (-itX_k)/2).
    # 
    #@-at
    #@@c

    #@<< Construct tensors to apply ZZ unitary >>
    #@+node:gmc.20080806124136.3:<< Construct tensors to apply ZZ unitary >>
    #@+at
    # To understand what is going on here, recall that
    # 
    #     exp(-it Z_1 Z_2) = exp(-it diag(+1,-1,-1,+1))
    #                      = diag(exp(-it),exp(+it),exp(+it),exp(-it))
    #                      = I_1 I_2 cos(t) - i Z_1 Z_2 sin(t)
    # 
    # Therefore, since the Z operators commute with each other,
    # 
    #     exp(sum_k i t Z_k Z_{k+1}) =
    #             prod_k (I_k I_{k+1} cos(t) - i Z_k Z_{k_1} sin(t))
    # 
    # Note that for each term, each site has a product of two operators:
    # one from k=n-1, and one from k=n.  Thus, depending on whether the
    # I_k I_{k+1} or the Z_k Z_{k+1} terms are chosen in these two factors,
    # there are four possible local operators that could appear at site n:
    # 
    #   I_n cos(t) (I_n chosen at both n-1 and n)
    # -iI_n sin(t) (Z_n chosen at both n-1 and n)
    #   Z_n cos(t) (Z_n chosen at n-1, I_n chosen at n)
    # -iZ_n sin(t) (I_n chosen at n-1, Z_n chosen at n)
    # 
    # Note that we have adopted the convention that the cos/sin will be
    # associated with the *right* of the two operators -- i.e., we have
    # expressed the sum above in the form
    # 
    #     exp(sum_k i t Z_k Z_{k+1}) =
    #             prod_k (I_k (I_{k+1} cos(t)) + Z_k (-iZ_{k_1} sin(t)))
    # 
    # (Observe the parantheses grouping the sin/cos with hte second operator.)
    # 
    # So the way our matrix product operator will work is that at each site,
    # it will choose to place either an II or a ZZ.  Of course, it can only
    # place one of these two operators at the site, so it sends a signal to
    # the right to tell its neighbor that it needs to multiply itself by
    # the second of these operators.  Likewise, this site receives a signal
    # from its left neighbor and based on that multiplies its choice of I or
    # Z with its left neighbors choice of either I cos(t) or -iZ sin(t).
    # 
    # On the left boundary, only a choice is made and I or Z placed.
    # 
    # On the right boundary, no choice is made;  it places either I cos(t)
    # or -iZ sin(t) based on the choice of its neighbor.
    # 
    # I use the signal 0 to indicate that an I was chosen, and the signal 1
    # to indicate that a Z was chosen.
    # 
    # So, having explained all that, we shall now construct the matrix product
    # operator representation of exp(sum_k -it Z_k Z_{k+1}).
    # First, the left boundary which makes a choice, placing either an I or a 
    # Z,
    # and then sends a signal to its right.  Specifically, we set up the
    # tensor left_operator_tensor so that
    # 
    #     left_operator_tensor[0] = I
    #     left_operator_tensor[1] = Z
    # 
    # using the following line of code:
    #@-at
    #@@c

    left_operator_tensor = array([I,Z])

    #@+at
    # Next, we shall set up the right boundary, which does not make a choice
    # itself but merely places either I cos(t) or -iZ sin(t) based on the 
    # choice
    # its neighbor made.  Specifically, we set up the tensor
    # right_operator_tensor so that
    # 
    #     right_operator_tensor[0] = I cos(dt)
    #     right_operator_tensor[1] = -iZ sin(dt)
    # 
    # using the following line of code:
    #@-at
    #@@c

    right_operator_tensor = array([I*cos(J*dt),miZ*sin(J*dt)])

    #@+at
    # Finally, we construct the interior tensors, which receives a signal from
    # the left telling it what it needs to multiply itself by, and sends a 
    # signal
    # to its right telling its neighbor what was chosen at this site.  
    # Specifically,
    # we set up interior_operator_tensor so that
    # 
    #     interior_operator_tensor[0,0] = I cos(dt)
    #     interior_operator_tensor[0,1] = Z cos(dt)
    #     interior_operator_tensor[1,0] = -iZ sin(dt)
    #     interior_operator_tensor[1,1] = -iI sin(dt)
    #@-at
    #@@c

    interior_operator_tensor = array([
    [I*cos(J*dt),Z*cos(J*dt)],
    [miZ*sin(J*dt),miI*sin(J*dt)]
    ])
    #@-node:gmc.20080806124136.3:<< Construct tensors to apply ZZ unitary >>
    #@nl

    #@<< Multiply by operators which apply the X unitary >>
    #@+node:gmc.20080806124136.4:<< Multiply by operators which apply the X unitary >>
    #@+at
    # At this point, we now have operator tensors which perform the unitary 
    # exp(-iZZ).
    # Obviously, we are not done yet since we still have to apply exp(-iX/2) 
    # on both
    # sides.  Recall that
    # 
    #     exp(-itX/2) = I cos(t/2) - iX sin(t/2)
    # 
    # We construct this matrix in the following line:
    #@-at
    #@@c

    expX = I*cos(dt/2) + miX*sin(dt/2)

    #@+at
    # Now we need to apply this operator to the left and right of the ZZ 
    # unitary.  To
    # do this, we multiply expX into the each of the last two indices of the 
    # operator
    # tensors, since these indices correspond to the physical observable.
    #@-at
    #@@c

    left_operator_tensor = multiply_tensor_by_matrix_at_index(left_operator_tensor,expX,1)
    left_operator_tensor = multiply_tensor_by_matrix_at_index(left_operator_tensor,expX,2)

    right_operator_tensor = multiply_tensor_by_matrix_at_index(right_operator_tensor,expX,1)
    right_operator_tensor = multiply_tensor_by_matrix_at_index(right_operator_tensor,expX,2)

    interior_operator_tensor = multiply_tensor_by_matrix_at_index(interior_operator_tensor,expX,2)
    interior_operator_tensor = multiply_tensor_by_matrix_at_index(interior_operator_tensor,expX,3)
    #@-node:gmc.20080806124136.4:<< Multiply by operators which apply the X unitary >>
    #@nl


    #@-node:gmc.20080805172037.80:<< Build MPO >>
    #@nl

    for dummy in xrange(number_of_time_steps_per_J):
        #@        << Apply unitary >>
        #@+node:gmc.20080805172037.75:<< Apply unitary >>
        #@+at
        # The tensors at the boundaries are a special case, so we handle them 
        # separately.
        # 
        # First, absorb the lambdas into the site tensors.  Note that the 
        # lambdas are
        # assumed to be shaped so that they get multiplied into the correct 
        # index.
        #@-at
        #@@c
        site_tensors[0] *= lambdas[0]
        site_tensors[1] *= lambdas[1]

        #@+at
        # Apply the MPO to these site tensors.
        #@-at
        #@@c
        site_tensors[0] = multiply_left_site_tensor_by_operator_tensor(site_tensors[0],left_operator_tensor)
        site_tensors[1] = multiply_interior_site_tensor_by_operator_tensor(site_tensors[1],interior_operator_tensor)

        #@+at
        # Reduce the size of the bond between the two tensors.
        #@-at
        #@@c
        site_tensors[0], lambda_, site_tensors[1] = merge_and_split(site_tensors[0],1,site_tensors[1],1,compressed_dimension)

        #@+at
        # Reshape the lambda so that it lines up with the correct index in
        # site_tensor[0].
        #@-at
        #@@c
        lambdas[0] = lambda_.reshape(1,lambda_.shape[0])

        #@+at
        # Now we handle the interior tensors.  This is essentially the same 
        # idea as before, so
        # step-by-step commentary will not be repeated.
        #@-at
        #@@c
        for i in xrange(1,number_of_sites-2):
            site_tensors[i+1] *= lambdas[i+1]
            site_tensors[i+1] = multiply_interior_site_tensor_by_operator_tensor(site_tensors[i+1],interior_operator_tensor)
            site_tensors[i], lambda_, site_tensors[i+1] = merge_and_split(site_tensors[i],2,site_tensors[i+1],1,compressed_dimension)
            lambdas[i] = lambda_.reshape(1,1,lambda_.shape[0])

        #@+at
        # Finally, we handle the right boundary.  Note that there is no lambda 
        # to the right
        # of the right boundary, and hence we skip the lambda absorption step.
        #@-at
        #@@c
        site_tensors[-1] = multiply_right_site_tensor_by_operator_tensor(site_tensors[-1],right_operator_tensor)
        site_tensors[-2], lambda_, site_tensors[-1] = merge_and_split(site_tensors[-2],2,site_tensors[-1],1,compressed_dimension)
        lambdas[-1] = lambda_.reshape(1,1,lambda_.shape[0])
        #@-node:gmc.20080805172037.75:<< Apply unitary >>
        #@nl

    if counter % (total_number_of_J_steps//10) == 0:
        print "\t%i/%i %f:%f" % (counter,total_number_of_J_steps,J,compute_energy(J))
    counter += 1
#@-node:gmc.20080805172037.76:<< Main loop >>
#@nl
#@-node:gmc.20080805172037.20:@thin time_evolution_transverse_ising.py
#@-leo
