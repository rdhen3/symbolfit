import uproot
import numpy as np
import itertools

# indir       = "/home/user1/Documents/PHS3350_Projects/ZeyuanJin/"
# filename    = "mB_vs_q2_constrained.root"


class Dataset:
    def __init__(self, x=None, y=None, bin_edges_2d=None, y_up=None, y_down=None):

        self.x = x
        self.y = y
        self.bin_edges_2d = bin_edges_2d
        self.y_up = y_up
        self.y_down = y_down


    def create_dataset_from_TH2D(self, indir, filename, cuts=None, blind=None):
        '''
        cuts : list
            List of length equal to the number of independent variables. 
            Each element is a list of (x_min, x_max) tuples defining ranges to cut
            in the variable x. The logical OR of the cuts is applied to the variable.
            e.g. cuts =  [[(x0_min1, x0_max1), (x0_min2, x0_max2)], [(x1_min1, x1_max1)]]
            will select bins that satisfy ( (x0_min1 <= x0 <= x0_max1) || (x0_min2 <= x0 <= x0_max2) ) && (x1_min1 <= x1 <= x1_max1)
        '''
        # Read in 2D histogram
        with uproot.open(f"{indir}/{filename}") as iF:
            h, x0_edges, x1_edges = iF["h_mBq2_cons"].to_numpy()

        h = np.maximum(h, 1e-1)

        if cuts is not None:
            x0cuts = [(x0_edges[:-1] >= cuts[0][c][0])*(x0_edges[1:] <= cuts[0][c][1]) for c in range(len(cuts[0]))]
            passx0 = list(itertools.accumulate(x0cuts, func = lambda a, b : a | b))[-1]
            # h           = h[(x0_edges[:-1] >= cuts[0][c][0])*(x0_edges[1:] <= cuts[0][c][1]), : ]
            h           = h[passx0, : ]
            x1cuts = [(x1_edges[:-1] >= cuts[1][c][0])*(x1_edges[1:] <= cuts[1][c][1]) for c in range(len(cuts[1]))]
            passx1 = list(itertools.accumulate(x1cuts, func = lambda a, b : a | b))[-1]
            # h           = h[ : , (x1_edges[:-1] >= cuts[1][c][0])*(x1_edges[1:] <= cuts[1][c][1])]
            h           = h[ : , passx1]
            # x0_edges     = x0_edges[(x0_edges >= cuts[0][0])*(x0_edges <= cuts[0][1])]
            # h           = h[ : , (x1_edges[:-1] >= cuts[1][c][0])*(x1_edges[1:] <= cuts[1][c][1])]
            # x1_edges     = x1_edges[(x1_edges >= cuts[1][0])*(x1_edges <= cuts[1][1])]
            x0cuts = [(x0_edges >= cuts[0][c][0])*(x0_edges <= cuts[0][c][1]) for c in range(len(cuts[0]))]
            passx0 = list(itertools.accumulate(x0cuts, func = lambda a, b : a | b))[-1]
            print(passx0)
            x0_edges     = x0_edges[passx0]
            x1cuts = [(x1_edges >= cuts[1][c][0])*(x1_edges <= cuts[1][c][1]) for c in range(len(cuts[1]))]
            passx1 = list(itertools.accumulate(x1cuts, func = lambda a, b : a | b))[-1]
            x1_edges     = x1_edges[passx1]

        x_centres = (x0_edges[:-1] + x0_edges[1:])/2
        y_centres = (x1_edges[:-1] + x1_edges[1:])/2

        if blind is not None:
            for b in blind:
                h[np.argmax(x0_edges[:-1] >= blind[b][0][0]):np.argmin(x0_edges[1:] <= blind[b][0][1]), 
                    np.argmax(x1_edges[:-1] >= blind[b][1][0]):np.argmin(x1_edges[1:] <= blind[b][1][1])] = np.inf
                # x0_edges     = x0_edges[(x0_edges < blind[0][0])|(x0_edges > blind[0][1])]
                # x1_edges     = x1_edges[(x1_edges < blind[1][0])|(x1_edges > blind[1][1])]



        # Dependent variable
        # Remove nan values from blinded regions
        h = h.flatten()
        h = h[~np.isnan(h)]
        self.y = h

        # Bin locations
        if blind is not None:
            x = [list(tup) for tup in itertools.product(list(x_centres), list(y_centres)) if any([tup[0]>blind[b][0][0] and tup[0]<blind[b][0][1] and tup[1]>blind[b][1][0] and tup[1]<blind[b][1][1] for b in range(len(blind))])]
        else:
            x = [list(tup) for tup in itertools.product(list(x_centres), list(y_centres))]

        self.x = x
        

        # Bin edges
        self.bin_edges_2d = [x0_edges.reshape(-1), x1_edges.reshape(-1)]


        # Bin errors
        self.y_up   = np.sqrt(self.y)
        self.y_down = np.sqrt(self.y)
