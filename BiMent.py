import scipy.stats
import scipy.special
import numpy as np
import time
import igraph as ig
from pprint import pprint
import matplotlib.pyplot as plt


def get_cooccurrence_edgelist(list_sets):
    '''
    Given a list of sets, compile the edgelist of the network 
    of co-occurrence between the symbols appearing in the sets.
    '''
    dict_edgelist = {}
    for s in list_sets:
        for x1 in s:
            for x2 in s:
                if x1 >= x2: continue
                try:
                    dict_edgelist[(x1, x2)] += 1
                except:
                    dict_edgelist[(x1, x2)] = 1
    return dict_edgelist


def get_graph(dict_edgelist):
    '''
    Given an edgelist dict, return an igraph
    Graph object.
    '''
    edgelist = [(key[0], key[1], value) for key, value in dict_edgelist.iteritems()]
    g = ig.Graph.TupleList(edgelist, weights=True)
    return g


def compute_significance(g, list_sets, solver_rms=1e-5, return_XY=False):
    '''
    Given a graph with an edge attribute 'weight'
    compute the 'significance' for each edge.
    @param g: the cooccurrence graph.
    @param list_sets: the list of sets from which
    the graph was compile. This is necessary to get
    the symbol frequencies.
    @param return_xy: whether or not to return X, Y, X_bak,
    Y_bak, fs, gs. If False (default), nothing is returned.
    '''
    dict_fs = get_symbol_frequencies(list_sets)
    list_gs = get_set_sizes(list_sets)

    N = float(sum(list_gs))

    names = dict_fs.keys()
    fs = dict_fs.values()
    # dict mapping node name (keys of dict_fs) to
    # their index in the fs array.

    dict_name_2_index_fs = {name: index for  index, name in enumerate(names)}

    # With this order, convert to array.
    fs = np.array(fs, dtype=np.float64)


    # We don't need a similar thing for the gs because they
    # already come ordered in a list, not a dict.
    gs = np.array(list_gs, dtype=np.float64)

    m, n = len(fs), len(gs)

    # Convert to arrays
    fs = fs.reshape(m, 1)
    gs = gs.reshape(n, 1)

    # Solve the saddle-point equations.
    X, Y, X_bak, Y_bak = solver(fs, gs , tolerance=solver_rms)

    # Attach additional attributes to the nodes
    # The frequency of symbol (f_i)
    g.vs['f'] = [ int(fs[dict_name_2_index_fs[v['name']]]) for v in g.vs]


    edgelist_significance = {}

    # Loop through the edges of the graph
    # and compute the pvalues.
    for e in g.es:
        ind1, ind2 = e.source, e.target
        name1, name2 = g.vs[ind1]['name'], g.vs[ind2]['name']

        # Index of the two end nodes in the X array
        i, j = dict_name_2_index_fs[name1], dict_name_2_index_fs[name2]

        # Probabilities of node i, j being connected to the sets
        ps_i = X[i] * Y / (1 + X[i] * Y)
        ps_j = X[j] * Y / (1 + X[j] * Y)
        ps = ps_i * ps_j

        mu = __mu(ps)
        sigma = __sigma(ps)
        eta = __eta(ps)

        weight = e['weight']
        logpvalue = logPvalue_RNA(weight, mu=mu, sigma=sigma, eta=eta)
        e['significance'] = -logpvalue if logpvalue < 0 else 0


#         pvalue = 1e-16 if pvalue <= 0 else pvalue
#         pvalue = 1. if pvalue > 1 else pvalue
#         print mu , sigma
#         e['pvalue'] = pvalue
#         e['significance'] = -np.log10(pvalue)

    if return_XY:
        return X, Y, X_bak, Y_bak, fs, gs


def get_symbol_frequencies(list_sets):
    '''
    Given a list of sets compute the frequencies
    of the symbols that appear in it.
    @return: dict mapping symbol to integer
    '''
    dict_frequencies = {}
    for s in list_sets:
        for x in s:
            try:
                dict_frequencies[x] += 1
            except:
                dict_frequencies[x] = 1
    return dict_frequencies


def get_set_sizes(list_sets):
    '''
    Given a list of sets, compute the sequence
    of the set sizes
    '''
    return [len(s) for s in list_sets]


# Numerical estimation of the Poisson binomial distribution cdf
def normal_pdf(x):
    '''Standard normal pdf'''
    return np.exp(-(x ** 2.) / 2.) / np.sqrt(2. * np.pi)


def normal_cdf(x):
    '''Srandard normal cdf'''
    return (1. + scipy.special.erf(x / np.sqrt(2.))) / 2.


def __sigma(ps):
    '''
    Compute the standard deviation for a Poisson binomial
    distribution.
    @param ps: numpy array of the independent probabilities
    '''
    return np.sqrt(np.sum(ps * (1. - ps)))


def __mu(ps):
    '''
    Compute the mean for a Poisson binomial
    distribution.
    @param ps: numpy array of the independent probabilities
    '''
    return ps.sum()


def __eta(ps):
    return np.sum(ps * (1. - ps) * (1. - 2. * ps))


def solver(fs, gs, tolerance=1e-10, max_iter=1000):
    '''
    Numerically solve the system of nonlinear equations
    we encounter when solving for the Lagrange multipliers

    @param fs: sequence of symbol frequencies.
    @param gs: sequence of set sizes.
    @param tolerance: solver continues iterating until the
    RMS of the difference between two consecutive solutions
    is less than tolerance.
    @param max_iter: maximum number of iterations.

    '''
    n, m = len(fs), len(gs)

    N = np.sum(fs)
    X = fs / np.sqrt(N)
    Y = gs / np.sqrt(N)

    X_bak = np.copy(X)
    Y_bak = np.copy(Y)

    x = fs * 0
    y = gs * 0

    # print fs, gs

    # Parameters
    change = 1
    max_iter = 1000
    t1 = time.time()
    for counter in range(max_iter):
        for i in xrange(n):
            x[i] = fs[i] / np.sum(Y / (1. + X[i] * Y))
        for i in xrange(m):
            y[i] = gs[i] / np.sum(X / (1. + X * Y[i]))

        
        # RMS
#         change = np.sqrt((np.sum((X - x) ** 2) + np.sum((Y - y) ** 2)) / (m + n))
        
        # L_oo
        change = max(np.max(np.abs(X - x))  , np.max(np.abs(Y - y)))

        X[:] = x
        Y[:] = y
        if change < tolerance: break
        #print counter, change
    t2 = time.time()
    print 'Solver done in {} seconds.'.format(round(t2 - t1), 2)

    if change > tolerance:
        raise Exception("Solver did not converge. Try increasing max_iter")
        
    print "Solver converged in {} iterations.".format(counter)
    return X, Y, X_bak, Y_bak



def RNA_cdf(k, **params):
    '''
    Estimate the CDF of the Poisson binomial distribution
    using the refined normal approximation of Volkova (1996).
    @param k: main argument of cdf function.
    @kwarg mu: mean.
    @kwarg sigma: standard deviation.
    @kwarg eta: intermediate parameter. See the paper.
    '''
    def __G(x, gamma):
        return normal_cdf(x) + gamma * (1. - x ** 2.) * normal_pdf(x) / 6.

    mu = params.get('mu')
    sigma = params.get('sigma')
    eta = params.get('eta')


    gamma = sigma ** (-3.) * eta
    cdf = __G((k + 0.5 - mu) / sigma, gamma)
    return cdf



def logPvalue_RNA(k, **params):
    mu = params.get('mu')
    sigma = params.get('sigma')
    eta = params.get('eta')
    gamma = sigma ** (-3.) * eta

    x = (k + 0.5 - mu) / sigma

    n_cdf = normal_cdf(x)

    if 1. - n_cdf < 1e-15:
        tmp = 1. - x ** 2.
        logpvalue = np.log10(gamma) + np.log10(np.abs(tmp))
        logpvalue += -np.log10(6) - 0.5 * np.log10(2 * np.pi)
        logpvalue += (-x ** 2. / 2.) * np.log10(np.e)
        if np.isnan(logpvalue):
            print x, gamma
    else:
        pvalue = 1. - (n_cdf + gamma * (1. - x ** 2.) * normal_pdf(x) / 6.)
        if pvalue < 0:
            print "Error pvalue less than 0"
        logpvalue = np.log10(pvalue)
    return logpvalue



def get_random_list_of_sets():
    list_sets = [set(list(np.random.randint(1,20, size = np.random.randint(1,5)))) for i in range(1000)]
    list_symbols_all = [x for a_set in list_sets for x in a_set] 
    return list_sets


def test():

    # Generate a random list of sets. Your data should be in the form
    # a list of Python Sets. A set can be a document, or an affiliation
    # and its elements are the words in the document, or the individuals
    # with the given affiliation. We have a list (ordered), so you can 
    # Keep track of which set is which, by keeping a separate list of
    # corresponding set labels. But the algorithm only needs this list
    # of sets. The set elements can be any Python objects, including numbers,
    # strings, etc.
    list_sets = get_random_list_of_sets()
    pprint(list_sets)

    # Given list_sets, compile the weighted edgelist of the co-occurrence
    # Graph. The result is a dict that maps each edge tuple to their integer
    # co-occurrence weight.
    dict_edgelist = get_cooccurrence_edgelist(list_sets)

    # Convert the edgelist to am igraph Graph instance.
    g = get_graph(dict_edgelist)

    # Print the graph's vertex and edge counts
    print g.vcount(), g.ecount()

    # For all edges in graph g, compute the significance. This value
    # is stored in the "significance" attribute for each edge. 
    compute_significance(g, list_sets, solver_rms=1e-6, return_XY=False)
    
    plt.plot(g.es['weight'], g.es['significance'],'.')
    plt.show()


if __name__ == "__main__":
    test()
